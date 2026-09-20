import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from ray_agent.shield_agent_protocol import (
    DeepSeekShieldToolAgent,
    ShieldAgentAction,
    ShieldAgentResponseError,
    validate_agent_action,
)
from ray_agent.shield_agent_tools import (
    ShieldAgentBudgets,
    ShieldToolEnvironment,
    ShieldToolError,
)
from ray_agent.shield_agent_verifier import (
    ShieldAgentVerifier,
    ShieldVerificationCriteria,
)
from ray_agent.shield_autonomous_agent import AutonomousShieldAgent
from ray_agent.shield_design import (
    ShieldDesign,
    ShieldEvaluation,
    baseline_lattice_design,
    calculate_shield_metrics,
    materialize_lattice_design,
    source_distant_mask,
    source_distant_sector_masks,
)


def _action(tool, arguments):
    return ShieldAgentAction(
        tool=tool,
        arguments=arguments,
        hypothesis=f"Test whether {tool} provides the next required evidence.",
        expected_observation="A quantitative result or an explicit gate rejection.",
        decision_reason="Use the next admissible instrument in the falsification chain.",
    )


class _ScaledFluxEvaluator:
    def __init__(self, fidelity: str, candidate_scale: float) -> None:
        self.fidelity = fidelity
        self.candidate_scale = float(candidate_scale)
        self.baseline_signature = baseline_lattice_design().signature

    def evaluate(self, design: ShieldDesign) -> ShieldEvaluation:
        problem = materialize_lattice_design(design, 14, 14)
        scale = 1.0 if design.signature == self.baseline_signature else self.candidate_scale
        flux = np.full((14, 14), scale, dtype=np.float64)
        far_mask = source_distant_mask(problem)
        metrics = calculate_shield_metrics(
            flux,
            far_mask,
            sector_masks=source_distant_sector_masks(problem, far_mask),
        )
        return ShieldEvaluation(
            design=design,
            flux=flux,
            metrics=metrics,
            ensemble_metrics=(metrics,),
            runtime_s=0.0,
            fidelity=self.fidelity,
        )


def _method_summary(histories, estimate):
    return {
        "histories": histories,
        "runtime_s": 1.0,
        "transported_particles_per_root": 2.0,
        "split_children": 10,
        "split_cap_hits": 0,
        "max_bank_size": 4,
        "regions": {
            "far_field": {
                "estimate": estimate,
                "standard_error": estimate * 0.03,
                "ci95": [estimate * 0.94, estimate * 1.06],
                "fom": 10.0,
            }
        },
    }


def _fake_mc_runner(
    baseline_design,
    candidate_design,
    baseline_guide_flux,
    candidate_guide_flux,
    *,
    histories,
    replicates,
    **kwargs,
):
    nx = int(kwargs.get("nx", 7))
    ny = int(kwargs.get("ny", 7))
    del baseline_guide_flux, candidate_guide_flux, kwargs
    total_histories = histories * replicates
    summary = {
        "grid": [nx, ny],
        "replicates": replicates,
        "designs": {
            "baseline": {
                "design": baseline_design.to_dict(),
                "analog": _method_summary(total_histories, 1.0),
                "variance_reduced": _method_summary(total_histories, 1.0),
                "unbiased_consistency": {
                    "far_field": {
                        "unpaired_consistency_z": 0.25,
                        "fom_gain": 3.0,
                    }
                },
            },
            "candidate": {
                "design": candidate_design.to_dict(),
                "analog": _method_summary(total_histories, 0.5),
                "variance_reduced": _method_summary(total_histories, 0.5),
                "unbiased_consistency": {
                    "far_field": {
                        "unpaired_consistency_z": -0.40,
                        "fom_gain": 4.0,
                    }
                },
            },
        },
        "design_comparison": {
            "far_field": {
                "candidate_over_baseline": 0.5,
                "estimated_reduction_fraction": 0.5,
                "difference": -0.5,
                "difference_z": -5.0,
                "candidate_upper_below_baseline_lower": True,
            },
            "deep_far_field": {
                "candidate_over_baseline": 0.7,
                "difference_z": -2.1,
                "candidate_upper_below_baseline_lower": False,
            },
        },
    }
    return summary, {"marker": np.ones((1,), dtype=np.float64)}


def _environment(
    tmp_path,
    budgets=None,
    *,
    require_projected_mc_audit=False,
    minimum_gmc_candidates_before_mc=1,
    minimum_gmc_guided_candidates_before_mc=0,
    discovery_mode="diffusion_prefilter",
    minimum_mc_histories_per_method=4,
    minimum_mc_replicates=1,
    minimum_mc_grid=7,
):
    return ShieldToolEnvironment(
        fast_evaluator=(
            None
            if discovery_mode == "gmc_only"
            else _ScaledFluxEvaluator("diffusion", 0.5)
        ),
        genvr_evaluator=_ScaledFluxEvaluator("genvr", 0.6),
        exact_evaluator=_ScaledFluxEvaluator("exact", 0.7),
        output_dir=tmp_path,
        budgets=budgets,
        metric_weights={"far_mean": 1.0},
        mc_runner=_fake_mc_runner,
        seed=17,
        require_projected_mc_audit=require_projected_mc_audit,
        minimum_gmc_candidates_before_mc=minimum_gmc_candidates_before_mc,
        minimum_gmc_guided_candidates_before_mc=(
            minimum_gmc_guided_candidates_before_mc
        ),
        discovery_mode=discovery_mode,
        minimum_mc_histories_per_method=minimum_mc_histories_per_method,
        minimum_mc_replicates=minimum_mc_replicates,
        minimum_mc_grid=minimum_mc_grid,
    )


def _new_candidate(environment):
    result = environment.execute(
        _action(
            "generate_structures",
            {"proposals_per_parent": 2, "beam_width": 1},
        )
    )
    return result["best_new_candidates"][0]["design_signature"]


def test_agent_action_rejects_protected_nested_fields():
    with pytest.raises(ValueError, match="protected field"):
        validate_agent_action(
            {
                "tool": "run_diffusion_batch",
                "arguments": {"operator_weights": {"source": 1.0}},
                "hypothesis": "Try to change a protected field.",
                "expected_observation": "The protocol rejects it.",
                "decision_reason": "Exercise the safety boundary.",
            }
        )
    with pytest.raises(ValueError, match="histories must be an integer"):
        validate_agent_action(
            {
                "tool": "run_unbiased_mc",
                "arguments": {
                    "design_signature": "candidate",
                    "histories": "many",
                },
                "hypothesis": "Attempt an invalid allocation.",
                "expected_observation": "The protocol rejects it.",
                "decision_reason": "Exercise typed tool arguments.",
            }
        )
    canonical = validate_agent_action(
        {
            "tool": "run_diffusion_batch",
            "arguments": {"proposals_per_parent": 2, "beam_width": 1},
            "hypothesis": "Exercise backward-compatible tool naming.",
            "expected_observation": "The action uses the canonical tool name.",
            "decision_reason": "Old trajectories must remain replayable.",
        }
    )
    assert canonical.tool == "generate_structures"


def test_gmc_discovery_config_controls_cli_defaults():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "25_run_autonomous_shield_agent.py"
    spec = importlib.util.spec_from_file_location("autonomous_shield_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous_argv = sys.argv
    try:
        sys.argv = [str(script)]
        arguments = module._parse_args()
        assert arguments.config == "configs/gmc_material_discovery_agent.json"
        assert arguments.max_steps == 20
        assert arguments.discovery_mode == "gmc_only"
        assert (arguments.gmc_nx, arguments.gmc_ny) == (112, 112)
        assert arguments.gmc_n_pos == 4
        assert module.gmc_state_count(4, 4, 32) == 512
        assert arguments.max_generated_structures == 96
        assert arguments.max_genvr_candidates == 16
        assert arguments.max_gmc_per_call == 4
        assert arguments.minimum_gmc_candidates_before_mc == 10
        assert arguments.minimum_gmc_guided_candidates_before_mc == 4
        assert arguments.minimum_mc_histories_per_method == 20_000
        assert arguments.minimum_mc_replicates == 4
        assert arguments.minimum_mc_grid == 56
        sys.argv.extend(["--max-steps", "9"])
        assert module._parse_args().max_steps == 9
    finally:
        sys.argv = previous_argv


def test_tool_environment_enforces_fidelity_order(tmp_path):
    environment = _environment(tmp_path)
    signature = _new_candidate(environment)
    with pytest.raises(ShieldToolError, match="GenVR/CFM gate"):
        environment.execute(
            _action("audit_projected_mc", {"design_signatures": [signature]})
        )
    with pytest.raises(ShieldToolError, match="prior GMC screening evidence"):
        environment.execute(
            _action(
                "certify_mc",
                {
                    "design_signature": signature,
                    "histories": 4,
                    "replicates": 1,
                    "nx": 7,
                    "ny": 7,
                },
            )
        )
    environment.execute(_action("screen_gmc", {"design_signatures": [signature]}))
    result = environment.execute(
        _action(
            "certify_mc",
            {
                "design_signature": signature,
                "histories": 4,
                "replicates": 1,
                "nx": 7,
                "ny": 7,
            },
        )
    )
    assert result["signature"] == signature
    with pytest.raises(ShieldToolError, match="only once"):
        environment.execute(
            _action(
                "certify_mc",
                {
                    "design_signature": signature,
                    "histories": 4,
                    "replicates": 1,
                    "nx": 7,
                    "ny": 7,
                },
            )
        )


def test_required_projected_mc_audit_remains_a_hard_gate(tmp_path):
    environment = _environment(tmp_path, require_projected_mc_audit=True)
    signature = _new_candidate(environment)
    environment.execute(_action("screen_gmc", {"design_signatures": [signature]}))

    with pytest.raises(ShieldToolError, match="requires a passed projected"):
        environment.execute(
            _action(
                "certify_mc",
                {
                    "design_signature": signature,
                    "histories": 4,
                    "replicates": 1,
                    "nx": 7,
                    "ny": 7,
                },
            )
        )

    environment.execute(
        _action("audit_projected_mc", {"design_signatures": [signature]})
    )
    result = environment.execute(
        _action(
            "certify_mc",
            {
                "design_signature": signature,
                "histories": 4,
                "replicates": 1,
                "nx": 7,
                "ny": 7,
            },
        )
    )
    assert result["signature"] == signature


def test_diffusion_budget_is_hard_limited(tmp_path):
    budgets = ShieldAgentBudgets(
        max_diffusion_evaluations=2,
        max_diffusion_per_call=1,
        max_genvr_candidates=1,
        max_exact_candidates=1,
        max_mc_root_histories=16,
        max_mc_histories_per_call=4,
        max_mc_replicates_per_call=1,
        max_mc_grid=7,
    )
    environment = _environment(tmp_path, budgets)
    result = environment.execute(
        _action(
            "run_diffusion_batch",
            {"proposals_per_parent": 1, "beam_width": 1},
        )
    )
    assert result["evaluated_candidates"] == 1
    assert environment.usage.diffusion_evaluations == 2
    with pytest.raises(ShieldToolError, match="exhausted"):
        environment.execute(
            _action(
                "run_diffusion_batch",
                {"proposals_per_parent": 1, "beam_width": 1},
            )
        )


def test_mc_requires_minimum_gmc_batch(tmp_path):
    budgets = ShieldAgentBudgets(
        max_diffusion_evaluations=6,
        max_diffusion_per_call=4,
        max_genvr_candidates=2,
        max_gmc_per_call=2,
        max_exact_candidates=1,
        max_mc_root_histories=16,
        max_mc_histories_per_call=4,
        max_mc_replicates_per_call=1,
        max_mc_grid=7,
    )
    environment = _environment(
        tmp_path,
        budgets,
        minimum_gmc_candidates_before_mc=2,
    )
    generated = environment.execute(
        _action(
            "generate_structures",
            {"proposals_per_parent": 2, "beam_width": 2},
        )
    )
    signatures = [
        record["design_signature"] for record in generated["best_new_candidates"]
    ]
    environment.execute(_action("screen_gmc", {"design_signatures": signatures[:1]}))
    with pytest.raises(ShieldToolError, match="minimum GMC discovery batch"):
        environment.execute(
            _action(
                "certify_mc",
                {
                    "design_signature": signatures[0],
                    "histories": 4,
                    "replicates": 1,
                    "nx": 7,
                    "ny": 7,
                },
            )
        )
    environment.execute(_action("screen_gmc", {"design_signatures": signatures[1:2]}))
    assert "certify_mc" in environment.executable_scientific_tools(4, 1)


def test_gmc_screened_parent_guides_and_unlocks_next_generation(tmp_path):
    budgets = ShieldAgentBudgets(
        max_diffusion_evaluations=8,
        max_diffusion_per_call=4,
        max_genvr_candidates=3,
        max_gmc_per_call=2,
        max_exact_candidates=1,
        max_mc_root_histories=16,
        max_mc_histories_per_call=4,
        max_mc_replicates_per_call=1,
        max_mc_grid=7,
    )
    environment = _environment(
        tmp_path,
        budgets,
        minimum_gmc_guided_candidates_before_mc=1,
    )
    parent_signature = _new_candidate(environment)
    environment.execute(
        _action("screen_gmc", {"design_signatures": [parent_signature]})
    )
    with pytest.raises(ShieldToolError, match="GMC-guided structures"):
        environment.execute(
            _action(
                "certify_mc",
                {
                    "design_signature": parent_signature,
                    "histories": 4,
                    "replicates": 1,
                    "nx": 7,
                    "ny": 7,
                },
            )
        )

    generated = environment.execute(
        _action(
            "generate_structures",
            {
                "parent_signatures": [parent_signature],
                "proposals_per_parent": 2,
                "beam_width": 2,
            },
        )
    )
    child_signature = generated["best_new_candidates"][0]["design_signature"]
    child = environment.candidate(child_signature)
    assert child is not None
    assert child.parent_signature == parent_signature
    assert child.generation_round == 2
    assert child.gmc_guided_generation is True
    assert child.generation_guidance_fidelity == "genvr"
    assert generated["gmc_guided_candidates_generated"] >= 1
    assert generated["best_new_candidates"][0]["operator_memory_source"] == (
        "gmc_validated"
    )
    assert generated["discovery_progress"]["gmc_guided_cycle_passed"] is False

    environment.execute(_action("screen_gmc", {"design_signatures": [child_signature]}))
    progress = environment.discovery_progress()
    assert progress["gmc_guided_and_screened"] == 1
    assert progress["mc_discovery_gate_passed"] is True
    assert "certify_mc" in environment.executable_scientific_tools(4, 1)
    assert sum(environment.gmc_operator_memory.counts.values()) == 2


def test_gmc_only_mode_generates_without_diffusion_and_enforces_precise_mc(tmp_path):
    budgets = ShieldAgentBudgets(
        max_diffusion_evaluations=8,
        max_diffusion_per_call=4,
        max_genvr_candidates=3,
        max_gmc_per_call=2,
        max_exact_candidates=1,
        max_mc_root_histories=32,
        max_mc_histories_per_call=4,
        max_mc_replicates_per_call=2,
        max_mc_grid=14,
    )
    environment = _environment(
        tmp_path,
        budgets,
        discovery_mode="gmc_only",
        minimum_gmc_candidates_before_mc=2,
        minimum_gmc_guided_candidates_before_mc=1,
        minimum_mc_histories_per_method=8,
        minimum_mc_replicates=2,
        minimum_mc_grid=14,
    )
    baseline = environment.candidate(environment.baseline_design.signature)
    assert baseline is not None and baseline.genvr is not None
    assert baseline.diffusion is None

    first = environment.execute(
        _action(
            "generate_structures",
            {"proposals_per_parent": 2, "beam_width": 1},
        )
    )
    parent_signature = first["best_new_candidates"][0]["design_signature"]
    parent = environment.candidate(parent_signature)
    assert parent is not None
    assert parent.diffusion is None
    assert parent.genvr is None
    assert parent.proposal_score is not None
    assert first["budget"]["used"] == 2

    first_screen = environment.execute(
        _action("screen_gmc", {"design_signatures": [parent_signature]})
    )
    visualization = first_screen["results"][0]["visualization"]
    image_path = tmp_path / visualization["geometry_and_gmc_field"]
    field_path = tmp_path / visualization["gmc_field"]
    assert image_path.exists() and image_path.stat().st_size > 0
    assert field_path.exists()
    with np.load(field_path) as fields:
        assert fields["gmc_flux"].shape == (14, 14)
        assert fields["geometry_mask"].shape == (7, 7)
    manifest = json.loads(
        (tmp_path / "gmc_screening" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["latest"]["signature"] == parent_signature
    assert (tmp_path / "gmc_screening" / "latest.png").exists()
    second = environment.execute(
        _action(
            "generate_structures",
            {
                "parent_signatures": [parent_signature],
                "proposals_per_parent": 2,
                "beam_width": 1,
            },
        )
    )
    child_signature = second["best_new_candidates"][0]["design_signature"]
    child = environment.candidate(child_signature)
    assert child is not None and child.gmc_guided_generation is True
    assert child.generation_guidance_fidelity == "genvr"
    environment.execute(
        _action("screen_gmc", {"design_signatures": [child_signature]})
    )
    assert environment.discovery_progress()["mc_discovery_gate_passed"] is True

    with pytest.raises(ShieldToolError, match="root histories per design and method"):
        environment.execute(
            _action(
                "certify_mc",
                {
                    "design_signature": child_signature,
                    "histories": 2,
                    "replicates": 2,
                    "nx": 14,
                    "ny": 14,
                },
            )
        )
    with pytest.raises(ShieldToolError, match="frozen range"):
        environment.execute(
            _action(
                "certify_mc",
                {
                    "design_signature": child_signature,
                    "histories": 4,
                    "replicates": 2,
                    "nx": 7,
                    "ny": 7,
                },
            )
        )
    result = environment.execute(
        _action(
            "certify_mc",
            {
                "design_signature": child_signature,
                "histories": 4,
                "replicates": 2,
                "nx": 14,
                "ny": 14,
            },
        )
    )
    assert result["signature"] == child_signature
    assert environment.budget_usage()["generated_structures"]["used"] == (
        first["generated_candidates"] + second["generated_candidates"]
    )


def test_agent_observation_is_compact_and_keeps_promoted_candidates(tmp_path):
    environment = _environment(tmp_path)
    signature = _new_candidate(environment)
    environment.execute(_action("promote_genvr", {"design_signatures": [signature]}))

    full = environment.observe()
    compact = environment.observe_for_agent(
        maximum_candidates=2,
        maximum_beam=1,
    )

    assert len(compact["visible_candidates"]) <= 2
    assert len(compact["beam"]) <= 1
    promoted = next(
        item for item in compact["visible_candidates"] if item["signature"] == signature
    )
    assert promoted["genvr"]["risk"] == pytest.approx(0.6)
    assert "metrics" not in promoted["genvr"]
    assert len(json.dumps(compact)) < len(json.dumps(full))


def test_verifier_rejects_premature_finish(tmp_path):
    environment = _environment(tmp_path)
    verifier = ShieldAgentVerifier(
        ShieldVerificationCriteria(
            minimum_histories_per_method=4,
            minimum_replicates=1,
        )
    )
    decision = verifier.review_finish(
        _action(
            "finish",
            {
                "candidate_signature": None,
                "claim": "inconclusive",
                "summary": "Stop before running any scientific tool.",
            },
        ),
        environment,
    )
    assert decision.accepted is False
    assert "scientific tools remain executable" in " ".join(decision.reasons)


class _ScriptedPolicy:
    def __init__(self):
        self.calls = 0
        self.first_signature = None
        self.signature = None
        self.last_trace = None

    def decide(self, observation, tools, memory):
        del tools, memory
        self.calls += 1
        if self.calls == 1:
            return _action(
                "finish",
                {
                    "candidate_signature": None,
                    "claim": "inconclusive",
                    "summary": "Premature probe of the verifier boundary.",
                },
            )
        if self.calls == 2:
            return _action(
                "generate_structures",
                {"proposals_per_parent": 2, "beam_width": 1},
            )
        candidates = [
            item
            for item in observation["visible_candidates"]
            if not item["is_baseline"]
        ]
        if self.calls == 3:
            self.first_signature = min(
                candidates,
                key=lambda item: item["diffusion"]["risk"],
            )["signature"]
            return _action(
                "screen_gmc",
                {"design_signatures": [self.first_signature]},
            )
        if self.calls == 4:
            return _action(
                "generate_structures",
                {
                    "parent_signatures": [self.first_signature],
                    "proposals_per_parent": 2,
                    "beam_width": 1,
                },
            )
        if self.calls == 5:
            guided_candidates = [
                item
                for item in candidates
                if item.get("lineage", {}).get("parent_signature")
                == self.first_signature
            ]
            self.signature = min(
                guided_candidates,
                key=lambda item: item["diffusion"]["risk"],
            )["signature"]
            return _action(
                "screen_gmc",
                {"design_signatures": [self.signature]},
            )
        if self.calls == 6:
            return _action(
                "certify_mc",
                {
                    "design_signature": self.signature,
                    "histories": 4,
                    "replicates": 1,
                    "nx": 7,
                    "ny": 7,
                    "n_levels": None,
                    "flux_exponent": 0.75,
                },
            )
        return _action(
            "finish",
            {
                "candidate_signature": self.signature,
                "claim": "verified_improvement",
                "summary": "GMC-guided regeneration and unbiased MC support the candidate.",
            },
        )


class _ListLogger:
    def __init__(self):
        self.records = []

    def write(self, event_type, payload):
        json.dumps(payload)
        self.records.append((event_type, payload))


class _RecoveringPolicy:
    def __init__(self):
        self.failed = False
        self.scripted = _ScriptedPolicy()
        self.last_trace = None

    def decide(self, observation, tools, memory):
        if not self.failed:
            self.failed = True
            raise ShieldAgentResponseError("simulated truncated JSON")
        return self.scripted.decide(observation, tools, memory)


def test_complete_autonomous_tool_loop_with_independent_verifier(tmp_path):
    budgets = ShieldAgentBudgets(
        max_diffusion_evaluations=8,
        max_diffusion_per_call=4,
        max_genvr_candidates=2,
        max_exact_candidates=1,
        max_mc_root_histories=32,
        max_mc_histories_per_call=8,
        max_mc_replicates_per_call=2,
        max_mc_grid=14,
    )
    environment = _environment(
        tmp_path,
        budgets,
        minimum_gmc_guided_candidates_before_mc=1,
    )
    verifier = ShieldAgentVerifier(
        ShieldVerificationCriteria(
            minimum_histories_per_method=4,
            minimum_replicates=1,
        )
    )
    logger = _ListLogger()
    result = AutonomousShieldAgent(
        policy=_ScriptedPolicy(),
        environment=environment,
        verifier=verifier,
        max_steps=7,
        logger=logger,
    ).run()

    assert result.final_claim == "verified_improvement"
    assert result.candidate_signature is not None
    assert result.verifier_decision.accepted is True
    assert result.termination == "verifier_accepted_finish"
    serialized = result.to_dict(environment)
    assert serialized["deterministic_policy_fallback"] is False
    assert serialized["final_design"]["signature"] == result.candidate_signature
    json.dumps(serialized)
    event_types = [event_type for event_type, _ in logger.records]
    for required in (
        "agent_input",
        "agent_output",
        "decision",
        "tool_call",
        "tool_result",
        "environment_feedback",
        "verifier_decision",
        "run_completed",
    ):
        assert required in event_types
    assert any(
        payload.get("type") == "finish_rejected"
        for event_type, payload in logger.records
        if event_type == "environment_feedback"
    )


def test_response_format_failure_does_not_consume_scientific_step(tmp_path):
    budgets = ShieldAgentBudgets(
        max_diffusion_evaluations=8,
        max_diffusion_per_call=4,
        max_genvr_candidates=2,
        max_exact_candidates=1,
        max_mc_root_histories=32,
        max_mc_histories_per_call=8,
        max_mc_replicates_per_call=2,
        max_mc_grid=14,
    )
    environment = _environment(
        tmp_path,
        budgets,
        minimum_gmc_guided_candidates_before_mc=1,
    )
    verifier = ShieldAgentVerifier(
        ShieldVerificationCriteria(
            minimum_histories_per_method=4,
            minimum_replicates=1,
        )
    )
    logger = _ListLogger()
    result = AutonomousShieldAgent(
        policy=_RecoveringPolicy(),
        environment=environment,
        verifier=verifier,
        max_steps=7,
        max_response_failures=2,
        logger=logger,
    ).run()

    assert result.final_claim == "verified_improvement"
    assert result.steps_completed == 7
    assert result.response_failures == 1
    first_step_inputs = [
        payload
        for event_type, payload in logger.records
        if event_type == "agent_input" and payload["step"] == 1
    ]
    assert len(first_step_inputs) == 2


class _FailingPolicy:
    last_trace = None

    def decide(self, observation, tools, memory):
        del observation, tools, memory
        raise RuntimeError("simulated API outage")


def test_policy_failure_aborts_without_deterministic_fallback(tmp_path):
    logger = _ListLogger()
    agent = AutonomousShieldAgent(
        policy=_FailingPolicy(),
        environment=_environment(tmp_path),
        verifier=ShieldAgentVerifier(
            ShieldVerificationCriteria(
                minimum_histories_per_method=4,
                minimum_replicates=1,
            )
        ),
        max_steps=2,
        logger=logger,
    )
    with pytest.raises(RuntimeError, match="simulated API outage"):
        agent.run()
    failures = [
        payload
        for event_type, payload in logger.records
        if event_type == "agent_failure"
    ]
    assert failures and failures[-1]["fallback_used"] is False


class _FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_deepseek_tool_agent_recovers_from_empty_reasoning_response(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    requests = []
    responses = [
        {
            "id": "first",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": ""},
                }
            ],
            "usage": {
                "completion_tokens": 1000,
                "completion_tokens_details": {"reasoning_tokens": 1000},
            },
        },
        {
            "id": "second",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "tool": "run_diffusion_batch",
                                "arguments": {
                                    "proposals_per_parent": 2,
                                    "beam_width": 1,
                                },
                                "hypothesis": "A legal swap may reduce far-field risk.",
                                "expected_observation": "The diffusion risk decreases.",
                                "decision_reason": "Establish candidates before promotion.",
                            }
                        )
                    },
                }
            ],
            "usage": {"completion_tokens": 120},
        },
    ]

    def fake_urlopen(request, timeout):
        del timeout
        requests.append(json.loads(request.data.decode("utf-8")))
        return _FakeHTTPResponse(responses.pop(0))

    monkeypatch.setattr(
        "ray_agent.shield_agent_protocol.urllib.request.urlopen",
        fake_urlopen,
    )
    policy = DeepSeekShieldToolAgent(
        thinking="enabled",
        reasoning_effort="high",
        max_tokens=1000,
        response_retries=1,
    )
    action = policy.decide(
        observation={"visible_candidates": []},
        tools=[{"name": "run_diffusion_batch", "arguments": {}}],
        memory=[],
    )

    assert action.tool == "generate_structures"
    assert len(policy.last_traces) == 2
    assert policy.last_traces[0]["accepted"] is False
    assert policy.last_traces[1]["accepted"] is True
    assert requests[0]["thinking"]["type"] == "enabled"
    assert requests[1]["thinking"]["type"] == "disabled"
    assert requests[1]["reasoning_effort"] == "low"
    assert requests[1]["max_tokens"] == 2048
