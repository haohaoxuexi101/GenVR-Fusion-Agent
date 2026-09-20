from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from gmc.checkpoints import (
    DEFAULT_BOUNDARY_CHECKPOINT,
    DEFAULT_INTERNAL_CHECKPOINT,
    load_cfm_checkpoint,
    require_recommended_path_mode,
)
from gmc.method_taxonomy import FLUX_FIELD_ALIASES, get_method_taxonomy
from ray_agent.metrics import load_fluxes, select_reference_flux
from ray_agent.reporting import finalize_run


ROOT = Path(__file__).resolve().parents[1]


def test_method_taxonomy_rejects_exact_truth_claims():
    taxonomy = get_method_taxonomy()
    assert taxonomy["legacy_flux_key_aliases"]["oracle"] == "projected_local_mc_response"
    assert set(FLUX_FIELD_ALIASES) <= set(taxonomy["methods"])
    assert all(method["continuum_truth"] is False for method in taxonomy["methods"].values())


def test_stage8_defaults_use_best_hpc_v2_checkpoints():
    script = ROOT / "scripts" / "30_stage8_ultra_benchmark.py"
    spec = importlib.util.spec_from_file_location("stage8_ultra_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args([])
    assert args.boundary_ckpt == DEFAULT_BOUNDARY_CHECKPOINT
    assert args.internal_ckpt == DEFAULT_INTERNAL_CHECKPOINT
    assert args.allow_legacy_checkpoints is False
    assert args.dense_spatial_scheme == "diamond"


def test_checkpoint_kind_and_path_mode_validation(tmp_path):
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "kind": "boundary",
            "model_kind": "boundary",
            "epoch": 3,
            "best_val_loss": 1.25,
            "transform": {},
            "model_state": {},
            "model_config": {"c_dim": 5, "y_dim": 4},
            "train_args": {"n_train": 10, "n_val": 2, "total_epochs": 3},
        },
        checkpoint_path,
    )
    _, metadata = load_cfm_checkpoint(checkpoint_path, "boundary")
    assert metadata["path_mode"] == "absolute"
    with pytest.raises(ValueError, match="allow-legacy-checkpoints"):
        require_recommended_path_mode(metadata)
    with pytest.raises(ValueError, match="expected a 'internal' checkpoint"):
        load_cfm_checkpoint(checkpoint_path, "internal")


def test_load_fluxes_supports_legacy_and_canonical_keys(tmp_path):
    cfm = np.arange(16, dtype=np.float64).reshape(4, 4) + 1.0
    dense = np.flipud(cfm)
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, cfm=cfm, oracle=np.array([]), dense=dense, mc=np.array([]))

    fluxes = load_fluxes(path)
    np.testing.assert_array_equal(fluxes["cfm"], fluxes["genvr_cfm_response"])
    np.testing.assert_array_equal(fluxes["dense"], fluxes["dense_sn_reference"])
    method_id, reference = select_reference_flux(
        fluxes,
        preference=("projected_local_mc_response", "dense_sn_reference"),
    )
    assert method_id == "dense_sn_reference"
    np.testing.assert_array_equal(reference, dense)


def test_load_fluxes_rejects_conflicting_aliases(tmp_path):
    path = tmp_path / "conflict.npz"
    np.savez_compressed(
        path,
        cfm=np.ones((3, 3)),
        genvr_cfm_response=np.zeros((3, 3)),
        dense=np.ones((3, 3)),
    )
    with pytest.raises(ValueError, match="conflicting NPZ aliases"):
        load_fluxes(path)


def test_reporting_labels_dense_fallback_correctly(tmp_path):
    yy, xx = np.mgrid[:16, :16]
    dense = 1.0 + xx + 0.3 * yy
    cfm = dense * (1.0 + 0.01 * np.sin(xx))
    flux_path = tmp_path / "fluxes.npz"
    np.savez_compressed(flux_path, cfm=cfm, oracle=np.array([]), dense=dense, mc=np.array([]))
    ray = {
        "residual_rms": 0.01,
        "directional_concentration": 0.2,
        "directional_peak_to_mean": 1.0,
        "ray_index": 0.002,
    }
    observation = {
        "experiment_id": "coarse",
        "n_state": 32,
        "process_elapsed_s": 0.1,
        "cfm_ray_vs_dense": ray,
        "cfm_vs_dense": {"corr": 0.99, "normalized_rel_l1": 0.01, "log10_rmse": 0.01},
        "artifacts": {"fluxes": flux_path.name},
    }

    summary = finalize_run(tmp_path, "lattice", [observation], [])
    assert summary["common_reference"] == "coarse:dense_sn_reference"
    assert "oracle_vs_common" not in summary["causal_experiments"][0]
    assert summary["ray_index_reduction_vs_fixed_dense"]["combined_oracle_percent"] is None
