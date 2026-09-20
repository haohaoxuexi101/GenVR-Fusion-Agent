from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import torch

from .metrics import diagnose_experiment, finite_json
from .scientific import build_controlled_comparisons
from .schema import AgentConfig, ExperimentAction


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TrajectoryLogger:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "timestamp_utc": utc_now(),
            "event_type": event_type,
            **finite_json(payload),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class RayEffectEnvironment:
    def __init__(self, root: Path, config: AgentConfig, run_dir: Path, run_id: str) -> None:
        self.root = root.resolve()
        self.config = config
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = self.run_dir / "experiments"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.logger = TrajectoryLogger(self.run_dir / "trajectory.jsonl", run_id)
        self.device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
        if self.device == "auto":
            self.device = "cpu"
        self.oracle_backend = "cuda" if self.device == "cuda" else "cpu"

    def write_manifest(self, config_path: Path) -> dict[str, Any]:
        tracked = [
            config_path,
            self.root / "scripts" / "run_ray_agent.py",
            self.root / "scripts" / "30_stage8_ultra_benchmark.py",
            self.root / "ray_agent" / "cli.py",
            self.root / "ray_agent" / "environment.py",
            self.root / "ray_agent" / "metrics.py",
            self.root / "ray_agent" / "policies.py",
            self.root / "ray_agent" / "reporting.py",
            self.root / "ray_agent" / "scientific.py",
            self.root / "ray_agent" / "schema.py",
            self.root / "gmc" / "checkpoints.py",
            self.root / "gmc" / "method_taxonomy.py",
            self.root / "gmc" / "response_matrix2d.py",
            self.root / "gmc" / "response_operator_ultra.py",
            self.root / self.config.boundary_ckpt,
            self.root / self.config.internal_ckpt,
        ]
        manifest = {
            "created_utc": utc_now(),
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "policy": self.config.policy,
            "human_intervention": "None after launch; action pool and stopping rule were predeclared in the config.",
            "selection_rule": (
                "The scientific policy closes missing one-factor contrasts before certification; "
                "all completed trials are reported and recommendations require declared gates."
                if self.config.policy == "scientific"
                else "All completed trials are reported; no unreported best-of-N filtering."
            ),
            "files": {},
        }
        for path in tracked:
            full = path if path.is_absolute() else self.root / path
            if full.exists() and full.is_file():
                manifest["files"][str(full.relative_to(self.root) if full.is_relative_to(self.root) else full)] = {
                    "bytes": full.stat().st_size,
                    "sha256": sha256_file(full),
                }
        (self.run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.logger.write("run_started", {"manifest": manifest, "config": self.config.to_dict()})
        return manifest

    def run_experiment(
        self,
        action: ExperimentAction,
        role: str,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        action.validate()
        outdir = self.experiment_dir / action.experiment_id
        outdir.mkdir(parents=True, exist_ok=True)
        cache_dir = self.config.response_cache_dir
        rebuild_args: list[str] = []
        if role == "random_reference":
            # A random-reference trial must not silently reuse a causal trial's
            # seed-specific response library.  Keep its cache isolated and force
            # construction so the recorded random seed actually takes effect.
            cache_dir = f"{self.config.response_cache_dir}/random/{action.experiment_id}"
            rebuild_args = ["--rebuild-response"]
        command = [
            sys.executable,
            str(self.root / "scripts" / "30_stage8_ultra_benchmark.py"),
            "--case", self.config.case,
            "--nx", str(self.config.nx),
            "--ny", str(self.config.ny),
            "--n-pos", str(action.n_pos),
            "--n-mu", str(action.n_mu),
            "--n-phi", str(action.n_phi),
            "--phi-offset-fraction", str(action.interface_phi_offset_fraction),
            "--cfm-mode", action.cfm_mode,
            "--boundary-ckpt", self.config.boundary_ckpt,
            "--internal-ckpt", self.config.internal_ckpt,
            "--cfm-samples", str(action.cfm_samples),
            "--oracle-samples", str(action.oracle_samples),
            "--internal-samples", str(action.internal_samples),
            "--oracle-internal-samples", str(action.oracle_internal_samples),
            "--mc-histories", str(
                self.config.mc_histories if action.mc_histories is None else action.mc_histories
            ),
            "--device", self.device,
            "--oracle-backend", self.oracle_backend,
            "--oracle-mc-dtype", self.config.oracle_mc_dtype,
            "--dtype", self.config.dtype,
            "--rk4-steps", str(self.config.rk4_steps),
            "--inference-batch-size", str(self.config.inference_batch_size),
            "--matmul-precision", "highest",
            "--solver", "auto",
            "--check-every", str(self.config.check_every),
            "--rtol", str(self.config.rtol),
            "--max-iters", str(self.config.max_iters),
            "--dense-n-mu", str(action.dense_n_mu or self.config.dense_n_mu),
            "--dense-n-phi", str(action.dense_n_phi or self.config.dense_n_phi),
            "--dense-phi-offset-fraction", str(action.dense_phi_offset_fraction),
            "--dense-rtol", str(self.config.dense_rtol),
            "--dense-max-iters", str(self.config.dense_max_iters),
            "--seed", str(action.seed),
            "--response-cache-dir", cache_dir,
            "--outdir", str(outdir),
        ] + rebuild_args
        self.logger.write("agent_input", {
            "role": role,
            "observation": "Current experiment history is available in prior trajectory events.",
            "research_question": (
                "Which numerical layer creates direction-locked artifacts, and which configuration "
                "minimizes rotation sensitivity without sacrificing projected-MC or global-MC fidelity?"
            ),
        })
        self.logger.write("agent_output", {"role": role, "hypothesis": action.hypothesis, "action": action.to_dict()})
        self.logger.write("tool_call", {"tool": "stage8_ultra_benchmark", "command": command, "cwd": str(self.root)})
        start = time.perf_counter()
        proc = subprocess.run(command, cwd=self.root, text=True, capture_output=True, encoding="utf-8", errors="replace")
        elapsed = time.perf_counter() - start
        (outdir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (outdir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
        self.logger.write("tool_result", {
            "tool": "stage8_ultra_benchmark",
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "stdout_file": str((outdir / "stdout.txt").relative_to(self.run_dir)),
            "stderr_file": str((outdir / "stderr.txt").relative_to(self.run_dir)),
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-3000:],
        })
        if proc.returncode != 0:
            self.logger.write("error", {"experiment_id": action.experiment_id, "message": "benchmark failed", "returncode": proc.returncode})
            raise RuntimeError(f"experiment {action.experiment_id} failed; see {outdir / 'stderr.txt'}")
        metrics_path = outdir / "metrics.json"
        npz_path = outdir / f"{self.config.case}_stage8_ultra.npz"
        observation = diagnose_experiment(npz_path, metrics_path, self.config.case)
        observation.update({
            "experiment_id": action.experiment_id,
            "role": role,
            "action": action.to_dict(),
            "process_elapsed_s": elapsed,
            "artifacts": {
                "metrics": str(metrics_path.relative_to(self.run_dir)),
                "fluxes": str(npz_path.relative_to(self.run_dir)),
                "figure": str((outdir / f"{self.config.case}_stage8_ultra.png").relative_to(self.run_dir)),
            },
        })
        comparisons = build_controlled_comparisons(
            self.run_dir,
            [*(history or []), observation],
        )
        observation["controlled_comparisons"] = [
            comparison
            for comparison in comparisons
            if observation["experiment_id"]
            in {
                comparison["baseline_experiment"],
                comparison["intervention_experiment"],
            }
        ]
        (outdir / "diagnostics.json").write_text(
            json.dumps(finite_json(observation), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.logger.write("environment_feedback", {"experiment_id": action.experiment_id, "observation": observation})
        return observation
