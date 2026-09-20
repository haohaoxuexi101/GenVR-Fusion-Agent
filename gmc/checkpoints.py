"""Checkpoint loading, validation and provenance helpers for CFM models."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import torch


DEFAULT_BOUNDARY_CHECKPOINT = "outputs/hpc_v2_models/boundary_900k_100ep/best.pt"
DEFAULT_INTERNAL_CHECKPOINT = "outputs/hpc_v2_models/internal_900k_100ep/best.pt"
RECOMMENDED_PATH_MODE = "excess_chord"


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cfm_checkpoint(
    checkpoint_path: str | Path,
    expected_kind: str,
    map_location: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a checkpoint and return it with auditable, JSON-safe metadata."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"CFM checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"CFM checkpoint must be a mapping: {path}")

    kind = checkpoint.get("kind", checkpoint.get("model_kind"))
    if kind != expected_kind:
        raise ValueError(f"expected a {expected_kind!r} checkpoint, got {kind!r}: {path}")
    if "transform" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError(f"checkpoint is missing transform/model_state: {path}")

    transform = checkpoint["transform"]
    if isinstance(transform, dict):
        path_mode = str(transform.get("path_mode", "absolute"))
    else:
        path_mode = str(getattr(transform, "path_mode", "absolute"))
    training = checkpoint.get("train_args") or checkpoint.get("train_config") or {}
    model_config = checkpoint.get("model_config", {})
    best_val_loss = checkpoint.get("best_val_loss")
    metadata = {
        "path": str(path),
        "sha1": _sha1(path),
        "kind": str(kind),
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "path_mode": path_mode,
        "n_train": int(training.get("n_train", -1)),
        "n_val": int(training.get("n_val", -1)),
        "total_epochs": int(training.get("total_epochs", -1)),
        "model_config": dict(model_config),
    }
    return checkpoint, metadata


def require_recommended_path_mode(metadata: dict[str, Any]) -> None:
    """Reject legacy absolute-path models in quality-first benchmark runs."""

    if metadata["path_mode"] != RECOMMENDED_PATH_MODE:
        raise ValueError(
            f"checkpoint {metadata['path']} uses path_mode={metadata['path_mode']!r}; "
            f"quality-first runs require {RECOMMENDED_PATH_MODE!r}. "
            "Pass --allow-legacy-checkpoints only for an explicit historical comparison."
        )
