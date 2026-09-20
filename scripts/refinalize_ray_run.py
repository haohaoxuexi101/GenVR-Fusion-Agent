from __future__ import annotations

from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ray_agent.reporting import finalize_run
from ray_agent.schema import AgentConfig


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate a RayLab-GMC summary from retained diagnostics.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "config.json"
    config = _load(config_path)
    agent_config = AgentConfig.load(config_path)
    causal = []
    for action in config["actions"]:
        diagnostics_path = run_dir / "experiments" / action["experiment_id"] / "diagnostics.json"
        if diagnostics_path.exists():
            causal.append(_load(diagnostics_path))
    random_dirs = sorted((run_dir / "experiments").glob("random_*"))
    random_observations = [_load(path / "diagnostics.json") for path in random_dirs]
    summary = finalize_run(
        run_dir,
        config["case"],
        causal,
        random_observations,
        config=agent_config,
    )
    source_paths = [
        ROOT / "ray_agent" / "reporting.py",
        ROOT / "ray_agent" / "scientific.py",
        ROOT / "scripts" / "refinalize_ray_run.py",
    ]
    artifact_paths = [run_dir / "summary.json", run_dir / "summary.md", run_dir / "ablation_summary.png"]
    generated_utc = datetime.now(timezone.utc).isoformat()
    postprocess = {
        "schema_version": "1.0",
        "generated_utc": generated_utc,
        "operation": "Regenerated derived summary artifacts from retained experiment diagnostics; raw diagnostics and solver outputs were not changed.",
        "source_files": {
            str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in source_paths
        },
        "artifacts": {
            str(path.relative_to(run_dir)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in artifact_paths
        },
    }
    (run_dir / "postprocess_manifest.json").write_text(
        json.dumps(postprocess, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_id = "unknown"
    trajectory_path = run_dir / "trajectory.jsonl"
    if trajectory_path.exists():
        first = json.loads(trajectory_path.read_text(encoding="utf-8").splitlines()[0])
        run_id = first.get("run_id", run_id)
        event = {
            "schema_version": "1.0",
            "run_id": run_id,
            "timestamp_utc": generated_utc,
            "event_type": "summary_regenerated",
            "reason": postprocess["operation"],
            "scientific_claim": summary["scientific_claim"],
            "postprocess_manifest": "postprocess_manifest.json",
        }
        with trajectory_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(json.dumps({"run_dir": str(run_dir), "scientific_claim": summary["scientific_claim"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
