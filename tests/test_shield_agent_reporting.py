import json

import numpy as np
from PIL import Image

from ray_agent.shield_agent_reporting import write_autonomous_shield_figures
from ray_agent.shield_design import baseline_lattice_design


def _metrics(scale: float) -> dict[str, float]:
    return {
        "far_mean": 1.0 * scale,
        "far_p90": 1.2 * scale,
        "far_cvar90": 1.4 * scale,
        "far_max": 1.8 * scale,
        "worst_sector_mean": 1.3 * scale,
        "hotspot_ratio": 2.0 * scale,
        "angular_imbalance": 1.1 * scale,
    }


def _region(estimate: float) -> dict[str, float | list[float]]:
    error = 0.05 * estimate
    return {
        "estimate": estimate,
        "standard_error": error,
        "ci95": [max(estimate - 1.96 * error, 0.0), estimate + 1.96 * error],
    }


def _method(far: float, deep: float) -> dict[str, object]:
    return {
        "histories": 1000,
        "transported_particles_per_root": 3.0,
        "split_cap_hits": 0,
        "regions": {
            "far_field": _region(far),
            "deep_far_field": _region(deep),
        },
    }


def test_autonomous_reporting_writes_full_figure_set(tmp_path):
    baseline = baseline_lattice_design()
    candidate = baseline.swap((5, 1), (4, 3))
    mc_dir = tmp_path / "mc" / f"01_{candidate.signature}"
    mc_dir.mkdir(parents=True)
    mc_summary = {
        "designs": {
            "baseline": {
                "variance_reduced": _method(1.0, 0.3),
                "unbiased_consistency": {
                    "far_field": {"unpaired_consistency_z": 0.2},
                    "deep_far_field": {"unpaired_consistency_z": 2.4},
                },
            },
            "candidate": {
                "variance_reduced": _method(0.4, 0.24),
                "unbiased_consistency": {
                    "far_field": {"unpaired_consistency_z": -0.3},
                    "deep_far_field": {"unpaired_consistency_z": 0.4},
                },
            },
        },
        "design_comparison": {
            "far_field": {
                "candidate_over_baseline": 0.4,
                "estimated_reduction_fraction": 0.6,
                "difference_z": -5.0,
                "candidate_upper_below_baseline_lower": True,
            },
            "deep_far_field": {
                "candidate_over_baseline": 0.8,
                "estimated_reduction_fraction": 0.2,
                "difference_z": -1.0,
                "candidate_upper_below_baseline_lower": False,
            },
        },
    }
    (mc_dir / "certification.json").write_text(json.dumps(mc_summary), encoding="utf-8")
    shape = (14, 14)
    y_grid, x_grid = np.mgrid[: shape[0], : shape[1]]
    base_flux = np.exp(-0.2 * np.hypot(x_grid - 7.0, y_grid - 7.0))
    candidate_flux = 0.5 * base_flux
    far_mask = np.hypot(x_grid - 7.0, y_grid - 7.0) >= 6.0
    deep_mask = far_mask & (x_grid < 4)
    arrays = {
        "baseline_guide_flux": base_flux,
        "candidate_guide_flux": candidate_flux,
        "baseline_importance_levels": np.arange(shape[0] * shape[1]).reshape(shape) % 8,
        "candidate_importance_levels": np.arange(shape[0] * shape[1]).reshape(shape) % 10,
        "baseline_far_mask": far_mask,
        "candidate_far_mask": far_mask,
        "baseline_deep_far_mask": deep_mask,
        "candidate_deep_far_mask": deep_mask,
        "baseline_analog_flux": base_flux * 1.05,
        "baseline_variance_reduced_flux": base_flux,
        "candidate_analog_flux": candidate_flux * 0.95,
        "candidate_variance_reduced_flux": candidate_flux,
        "baseline_analog_relative_error": np.full(shape, 0.4),
        "baseline_variance_reduced_relative_error": np.full(shape, 0.1),
        "candidate_analog_relative_error": np.full(shape, 0.5),
        "candidate_variance_reduced_relative_error": np.full(shape, 0.12),
        "baseline_analog_cell_visits": np.full(shape, 4),
        "baseline_variance_reduced_cell_visits": np.full(shape, 12),
        "candidate_analog_cell_visits": np.full(shape, 3),
        "candidate_variance_reduced_cell_visits": np.full(shape, 10),
        "baseline_variance_reduced_split_children_map": np.full(shape, 6),
        "candidate_variance_reduced_split_children_map": np.full(shape, 5),
    }
    np.savez_compressed(mc_dir / "fields.npz", **arrays)
    summary = {
        "discovery_mode": "gmc_only",
        "final_claim": "verified_improvement",
        "candidate_signature": candidate.signature,
        "termination": "verifier_accepted_finish",
        "steps_completed": 5,
        "budget_usage": {
            "generated_structures": {"used": 12, "limit": 100},
            "genvr_candidates": {"used": 2, "limit": 4},
            "exact_candidates": {"used": 1, "limit": 2},
            "mc_root_histories": {"used": 4000, "limit": 8000},
        },
        "gmc_operator_memory": {
            "counts": {"gmc_path_importance": 8, "random_explore": 4},
            "mean_improvement": {"gmc_path_importance": 0.3, "random_explore": -0.1},
        },
        "discovery_progress": {
            "generated_structures": 12,
            "gmc_screened_structures": 2,
            "gmc_guided_and_screened": 1,
        },
        "verifier_decision": {
            "checks": {
                "genvr_gate": {"passed": True},
                "exact_gate": {"passed": True},
                "mc_gate": {"passed": True},
            }
        },
        "search_records": [
            {
                "design_signature": candidate.signature,
                "generation_round": 1,
                "gmc_risk": 0.6,
                "mutation": {"operator": "gmc_path_importance"},
            },
            {
                "design_signature": "random",
                "generation_round": 2,
                "gmc_risk": 0.8,
                "mutation": {"operator": "random_explore"},
            },
        ],
        "candidate_archive": [
            {
                "signature": baseline.signature,
                "is_baseline": True,
                "highest_fidelity": "genvr-cfm-response",
                "genvr": {"risk": 1.0, "metrics": _metrics(1.0)},
            },
            {
                "signature": candidate.signature,
                "is_baseline": False,
                "removed_from_baseline": [[5, 1]],
                "added_to_baseline": [[4, 3]],
                "highest_fidelity": "unbiased-mc",
                "genvr": {"risk": 0.6, "metrics": _metrics(0.6)},
                "exact": {"risk": 0.7, "metrics": _metrics(0.7)},
                "latest_mc": {
                    "artifact": {
                        "summary": str(mc_dir / "certification.json"),
                        "fields": str(mc_dir / "fields.npz"),
                    }
                },
            },
        ],
    }
    trajectory = tmp_path / "trajectory.jsonl"
    events = [
        {
            "event_type": "agent_output",
            "step": 1,
            "accepted": True,
            "action": {
                "tool": "run_diffusion_batch",
                "hypothesis": "Move weak absorbers toward high sensitivity cells.",
                "expected_observation": "Lower robust diffusion risk.",
            },
        },
        {
            "event_type": "tool_result",
            "step": 2,
            "result": {"stage": "genvr", "baseline_metrics": _metrics(1.0)},
        },
        {
            "event_type": "tool_result",
            "step": 3,
            "result": {"stage": "exact", "baseline_metrics": _metrics(1.0)},
        },
        {
            "event_type": "agent_output",
            "step": 5,
            "accepted": True,
            "action": {
                "tool": "finish",
                "hypothesis": "The candidate passes all independent gates.",
                "expected_observation": "The verifier accepts the claim.",
            },
        },
    ]
    trajectory.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    figures = write_autonomous_shield_figures(
        summary,
        tmp_path,
        trajectory_path=trajectory,
        response_fields={
            "baseline_genvr_cfm_response": base_flux * 0.98,
            "candidate_genvr_cfm_response": candidate_flux * 1.02,
            "baseline_projected_local_mc_response": base_flux,
            "candidate_projected_local_mc_response": candidate_flux,
            "metadata": {
                "candidate_signature": candidate.signature,
                "discretization": {
                    "n_pos": 2,
                    "n_mu": 4,
                    "n_phi": 32,
                    "n_angle": 32,
                    "n_state": 256,
                },
            },
        },
        dpi=45,
    )

    assert set(figures) == {
        "search_dashboard",
        "decision_timeline",
        "candidate_evidence",
        "gmc_discovery_storyboard",
        "gmc_screening_animation",
        "response_comparison",
        "verified_design",
        "mc_certification",
        "mc_population_control",
    }
    assert all((tmp_path / path.split("/")[-1]).exists() for path in figures.values())
    assert (tmp_path / "response_fields.npz").exists()
    assert (tmp_path / "response_fields.json").exists()
    with Image.open(tmp_path / "gmc_screening_reel.gif") as animation:
        assert animation.n_frames >= 3
    assert (tmp_path / "verified_design.png").read_bytes() != (
        tmp_path / "candidate_evidence_chain.png"
    ).read_bytes()
