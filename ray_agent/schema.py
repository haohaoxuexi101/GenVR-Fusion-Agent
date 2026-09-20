from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class ExperimentAction:
    experiment_id: str
    hypothesis: str
    n_pos: int
    n_mu: int
    n_phi: int
    cfm_samples: int
    oracle_samples: int
    internal_samples: int
    oracle_internal_samples: int
    seed: int
    cfm_mode: str = "direct"
    interface_phi_offset_fraction: float = 0.0
    dense_n_mu: int | None = None
    dense_n_phi: int | None = None
    dense_phi_offset_fraction: float = 0.0
    mc_histories: int | None = None

    def validate(self) -> None:
        if not self.experiment_id or any(c in self.experiment_id for c in "\\/:*?\"<>|"):
            raise ValueError(f"invalid experiment_id={self.experiment_id!r}")
        if self.n_pos < 1:
            raise ValueError("n_pos must be >= 1")
        if self.n_mu < 2:
            raise ValueError("n_mu must be >= 2")
        if self.n_phi < 8 or self.n_phi % 4:
            raise ValueError("n_phi must be >= 8 and divisible by 4")
        if min(self.cfm_samples, self.oracle_samples, self.internal_samples, self.oracle_internal_samples) < 1:
            raise ValueError("all sample counts must be positive")
        if self.cfm_mode not in {"direct", "ballistic-split"}:
            raise ValueError(f"unsupported cfm_mode={self.cfm_mode}")
        if not 0.0 <= self.interface_phi_offset_fraction < 1.0:
            raise ValueError("interface_phi_offset_fraction must be in [0,1)")
        if not 0.0 <= self.dense_phi_offset_fraction < 1.0:
            raise ValueError("dense_phi_offset_fraction must be in [0,1)")
        if self.dense_n_mu is not None and self.dense_n_mu < 2:
            raise ValueError("dense_n_mu must be >= 2")
        if self.dense_n_phi is not None and self.dense_n_phi < 4:
            raise ValueError("dense_n_phi must be >= 4")
        if self.mc_histories is not None and self.mc_histories < 0:
            raise ValueError("mc_histories must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ExperimentAction":
        action = ExperimentAction(**data)
        action.validate()
        return action


@dataclass(frozen=True)
class AgentConfig:
    run_name: str
    case: str
    nx: int
    ny: int
    boundary_ckpt: str
    internal_ckpt: str
    device: str
    dtype: str
    oracle_mc_dtype: str
    inference_batch_size: int
    rk4_steps: int
    rtol: float
    max_iters: int
    dense_n_mu: int
    dense_n_phi: int
    dense_rtol: float
    dense_max_iters: int
    check_every: int
    response_cache_dir: str
    mc_histories: int
    policy: str
    llm_model: str
    llm_thinking: str
    llm_reasoning_effort: str
    llm_max_tokens: int
    llm_timeout_s: int
    llm_required: bool
    max_causal_trials: int
    random_baseline_trials: int
    actions: tuple[ExperimentAction, ...]
    learning_shape_l1_limit: float = 0.08
    rotation_shape_l1_limit: float = 0.03
    angular_convergence_shape_l1_limit: float = 0.03
    repeatability_shape_l1_limit: float = 0.03
    global_mc_integral_rel_limit: float = 0.10
    require_global_mc_certification: bool = False
    require_angular_convergence: bool = False
    require_repeatability_for_attribution: bool = False
    require_candidate_repeatability: bool = False

    def validate(self) -> None:
        if self.case not in {"lattice", "hohlraum"}:
            raise ValueError("ray-agent currently supports lattice or hohlraum")
        if min(self.nx, self.ny) < 8:
            raise ValueError("nx and ny must be >= 8")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.policy not in {"causal", "scientific", "deepseek"}:
            raise ValueError("policy must be causal, scientific, or deepseek")
        if self.llm_thinking not in {"enabled", "disabled"}:
            raise ValueError("llm_thinking must be enabled or disabled")
        if self.llm_reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("llm_reasoning_effort must be low, high, or max")
        if self.llm_max_tokens < 64 or self.llm_timeout_s < 10:
            raise ValueError("LLM token and timeout budgets are too small")
        if len(self.actions) < 2:
            raise ValueError("at least two actions are required")
        if not 2 <= self.max_causal_trials <= len(self.actions):
            raise ValueError("max_causal_trials must be between 2 and the number of actions")
        if self.llm_required and self.policy != "deepseek":
            raise ValueError("llm_required=true requires policy=deepseek")
        if min(
            self.learning_shape_l1_limit,
            self.rotation_shape_l1_limit,
            self.angular_convergence_shape_l1_limit,
            self.repeatability_shape_l1_limit,
            self.global_mc_integral_rel_limit,
        ) <= 0.0:
            raise ValueError("scientific acceptance limits must be positive")
        ids = [a.experiment_id for a in self.actions]
        if len(ids) != len(set(ids)):
            raise ValueError("experiment_id values must be unique")
        for action in self.actions:
            action.validate()

    @staticmethod
    def load(path: str | Path) -> "AgentConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["actions"] = tuple(ExperimentAction.from_dict(x) for x in data["actions"])
        cfg = AgentConfig(**data)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["actions"] = [a.to_dict() for a in self.actions]
        return data
