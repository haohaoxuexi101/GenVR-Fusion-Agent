import numpy as np
import json
from dataclasses import replace
from pathlib import Path

from ray_agent.metrics import spectral_ray_metrics
from ray_agent.policies import CausalPolicy, DeepSeekPolicy, ScientificPolicy
from ray_agent.scientific import attribute_artifacts, build_controlled_comparisons
from ray_agent.schema import ExperimentAction


def _action(name: str) -> ExperimentAction:
    return ExperimentAction(name, "test", 1, 2, 16, 8, 8, 8, 8, 1)


def test_spectral_ray_metric_detects_directional_residual():
    n = 64
    reference = np.ones((n, n), dtype=np.float64)
    stripe = reference.copy()
    for i in range(n):
        stripe[i, i] += 0.5
        stripe[i, n - 1 - i] += 0.5
    isotropic_rng = np.random.default_rng(7)
    isotropic = reference + 0.04 * isotropic_rng.normal(size=(n, n))
    directional = spectral_ray_metrics(stripe, reference)
    noise = spectral_ray_metrics(isotropic, reference)
    assert directional["directional_concentration"] > noise["directional_concentration"]
    assert directional["ray_index"] > 0.0


def test_causal_policy_preserves_declared_order():
    actions = [_action("coarse"), _action("refined")]
    policy = CausalPolicy()
    selected, _ = policy.choose(actions, [])
    assert selected.experiment_id == "coarse"


def test_scientific_policy_prioritizes_missing_repeatability_contrast():
    baseline_action = _action("coarse")
    baseline = {
        "experiment_id": "coarse",
        "action": baseline_action.to_dict(),
        "dense_n_mu": 4,
        "dense_n_phi": 16,
    }
    angular = ExperimentAction("angular", "test", 1, 2, 32, 8, 8, 8, 8, 1)
    repeat = ExperimentAction("repeat", "test", 1, 2, 16, 8, 8, 8, 8, 2)

    selected, reason = ScientificPolicy().choose([angular, repeat], [baseline])

    assert selected.experiment_id == "repeat"
    assert "seed contrast" in reason


def _screening_comparison(kind, baseline, intervention, value):
    return {
        "comparison_type": kind,
        "baseline_experiment": baseline,
        "intervention_experiment": intervention,
        "method_sensitivity": {
            "cfm": {"shape": {"normalized_rel_l1": value}},
        },
    }


def _screening_observation(action, learning, comparisons=()):
    return {
        "experiment_id": action.experiment_id,
        "action": action.to_dict(),
        "dense_n_mu": 4,
        "dense_n_phi": 32,
        "cfm_learning_error": {"shape": {"normalized_rel_l1": learning}},
        "cfm_ray_vs_dense": {"ray_index": 0.0},
        "controlled_comparisons": list(comparisons),
    }


def test_scientific_policy_adaptively_selects_best_screened_certification():
    coarse = _action("coarse")
    coarse_rotation = replace(
        coarse,
        experiment_id="coarse_rotation",
        interface_phi_offset_fraction=0.25,
    )
    coarse_repeat = replace(coarse, experiment_id="coarse_repeat", seed=2)
    refined = replace(coarse, experiment_id="refined", n_phi=32)
    refined_rotation = replace(
        refined,
        experiment_id="refined_rotation",
        interface_phi_offset_fraction=0.25,
    )
    refined_repeat = replace(refined, experiment_id="refined_repeat", seed=2)
    coarse_certification = replace(
        coarse,
        experiment_id="coarse_certification",
        mc_histories=2000,
    )
    refined_certification = replace(
        refined,
        experiment_id="refined_certification",
        mc_histories=2000,
    )
    comparisons = [
        _screening_comparison("interface_rotation", "coarse", "coarse_rotation", 0.08),
        _screening_comparison("repeatability", "coarse", "coarse_repeat", 0.04),
        _screening_comparison("interface_rotation", "refined", "refined_rotation", 0.02),
        _screening_comparison("repeatability", "refined", "refined_repeat", 0.01),
        _screening_comparison("interface_angular_refinement", "coarse", "refined", 0.025),
    ]
    history = [
        _screening_observation(coarse, 0.06, comparisons),
        _screening_observation(coarse_rotation, 0.07),
        _screening_observation(coarse_repeat, 0.06),
        _screening_observation(refined, 0.03),
        _screening_observation(refined_rotation, 0.03),
        _screening_observation(refined_repeat, 0.03),
    ]

    selected, reason = ScientificPolicy(dense_n_mu=4, dense_n_phi=32).choose(
        [coarse_certification, refined_certification],
        history,
    )

    assert selected.experiment_id == "refined_certification"
    assert "mc_histories contrast" in reason


def test_scientific_policy_stops_only_after_all_certification_gates_pass():
    baseline = _action("candidate")
    rotation = replace(
        baseline,
        experiment_id="candidate_rotation",
        interface_phi_offset_fraction=0.25,
    )
    repeat = replace(baseline, experiment_id="candidate_repeat", seed=2)
    angular = replace(baseline, experiment_id="angular_reference", n_phi=32)
    certification = replace(
        baseline,
        experiment_id="candidate_certification",
        mc_histories=2000,
    )
    comparisons = [
        _screening_comparison("interface_rotation", "candidate", "candidate_rotation", 0.02),
        _screening_comparison("repeatability", "candidate", "candidate_repeat", 0.01),
        _screening_comparison("interface_angular_refinement", "candidate", "angular_reference", 0.02),
    ]
    history = [
        _screening_observation(baseline, 0.03, comparisons),
        _screening_observation(rotation, 0.03),
        _screening_observation(repeat, 0.03),
        _screening_observation(angular, 0.02),
    ]
    certified = _screening_observation(certification, 0.03)
    certified["cfm_vs_global_mc"] = {"level": {"integral_ratio": 1.05}}
    policy = ScientificPolicy(dense_n_mu=4, dense_n_phi=32)

    assert policy.should_stop(history) is False
    assert policy.should_stop(history + [certified]) is True
    certified["cfm_vs_global_mc"]["level"]["integral_ratio"] = 1.5
    assert policy.should_stop(history + [certified]) is False


def test_deepseek_policy_enforces_allowlist(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({
                "id": "test-response",
                "system_fingerprint": "test-fingerprint",
                "choices": [{"message": {"content": json.dumps({
                    "experiment_id": "refined",
                    "hypothesis": "test",
                    "decision_reason": "lower directional resolution is the next controlled probe",
                })}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }).encode()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-placeholder")
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    policy = DeepSeekPolicy(model="deepseek-v4-flash")
    baseline = {
        "experiment_id": "coarse",
        "n_state": 32,
        "cfm_ray_vs_dense": {"ray_index": 0.004},
        "cfm_vs_dense": {"normalized_rel_l1": 0.08},
        "wall_s": 2.0,
    }
    selected, reason = policy.choose([_action("refined")], [baseline])
    assert selected.experiment_id == "refined"
    assert "controlled probe" in reason
    assert policy.last_trace["secret_recorded"] is False


def test_controlled_rotations_attribute_dense_sn_sensitivity(tmp_path: Path):
    yy, xx = np.mgrid[:32, :32]
    base = 1.0 + 0.02 * xx + 0.01 * yy
    weak = base.copy()
    strong = base.copy()
    for index in range(32):
        weak[index, index] += 0.01
        strong[index, index] += 0.20

    def observation(name, interface_offset, dense_offset, cfm, dense):
        path = tmp_path / f"{name}.npz"
        np.savez_compressed(
            path,
            genvr_cfm_response=cfm,
            projected_local_mc_response=base,
            dense_sn_reference=dense,
            global_history_mc=np.array([]),
        )
        return {
            "experiment_id": name,
            "n_state": 32,
            "dense_n_mu": 2,
            "dense_n_phi": 16,
            "action": {
                "n_pos": 1,
                "n_mu": 2,
                "n_phi": 16,
                "interface_phi_offset_fraction": interface_offset,
                "dense_phi_offset_fraction": dense_offset,
                "cfm_samples": 64,
                "oracle_samples": 64,
                "internal_samples": 128,
                "oracle_internal_samples": 128,
                "seed": 1,
                "cfm_mode": "direct",
            },
            "artifacts": {"fluxes": path.name},
        }

    baseline = observation("baseline", 0.0, 0.0, base, base)
    interface_rotation = observation("interface_rotation", 0.25, 0.0, weak, base)
    dense_rotation = observation("dense_rotation", 0.0, 0.25, base, strong)
    comparisons = build_controlled_comparisons(
        tmp_path,
        [baseline, interface_rotation, dense_rotation],
    )
    attribution = attribute_artifacts(comparisons)

    assert {item["comparison_type"] for item in comparisons} == {
        "interface_rotation",
        "dense_rotation",
    }
    assert attribution["conclusion"] == "dense_sn_is_more_rotation_sensitive"
