import numpy as np
import pytest

from gmc.benchmarks2d import make_lattice_problem
from ray_agent.shield_design import (
    DiffusionShieldEvaluator,
    PhysicsGuidedProposalModel,
    ShieldDesign,
    ShieldEvaluation,
    SmartShieldAgent,
    baseline_lattice_design,
    calculate_shield_metrics,
    materialize_lattice_design,
    source_distant_mask,
    source_distant_sector_masks,
)
from ray_agent.shield_policy import EvidenceDrivenShieldStrategist, validate_shield_strategy


def test_baseline_design_reproduces_lattice_material_map():
    expected = make_lattice_problem(28, 28)
    generated = materialize_lattice_design(baseline_lattice_design(), 28, 28)
    assert np.array_equal(generated.sigma_a, expected.sigma_a)
    assert np.array_equal(generated.sigma_s, expected.sigma_s)
    assert generated.source_box == expected.source_box


def test_swap_preserves_mass_and_source_exclusion():
    baseline = baseline_lattice_design()
    candidate = baseline.swap((5, 1), (0, 0))
    assert candidate.absorber_count == baseline.absorber_count == 11
    assert (5, 1) not in candidate.absorbers
    assert (0, 0) in candidate.absorbers
    with pytest.raises(ValueError):
        baseline.swap((5, 1), baseline.source_cell)


def test_far_field_sectors_partition_the_far_mask():
    problem = make_lattice_problem(28, 28)
    far_mask = source_distant_mask(problem)
    sectors = source_distant_sector_masks(problem, far_mask)
    coverage = np.sum(np.stack(sectors, axis=0), axis=0)
    assert np.array_equal(coverage > 0, far_mask)
    assert np.all(coverage[far_mask] == 1)
    assert all(np.any(mask) for mask in sectors)


def test_metrics_report_angular_leakage():
    problem = make_lattice_problem(28, 28)
    far_mask = source_distant_mask(problem)
    sectors = source_distant_sector_masks(problem, far_mask)
    flux = np.ones((problem.ny, problem.nx), dtype=np.float64)
    flux[sectors[0]] = 3.0
    metrics = calculate_shield_metrics(flux, far_mask, sector_masks=sectors)
    assert metrics.worst_sector_mean == pytest.approx(3.0)
    assert metrics.angular_imbalance > 1.0


def test_strategy_validation_accepts_only_compiled_controls():
    strategy = validate_shield_strategy(
        {
            "operator_weights": {
                "adjoint_sensitivity": 1.0,
                "channel_block": 0.5,
            },
            "focus_sector": 3,
            "exploration_fraction": 0.15,
            "hypothesis": "Test the measured high-sensitivity sector.",
        }
    )
    assert strategy.focus_sector == 3
    assert strategy.normalized_operator_weights()["adjoint_sensitivity"] > 0.0
    with pytest.raises(ValueError):
        validate_shield_strategy(
            {
                "operator_weights": {"invent_geometry": 1.0},
                "focus_sector": 0,
                "exploration_fraction": 0.2,
            }
        )


def test_evidence_strategist_uses_measured_hot_sector():
    strategy = EvidenceDrivenShieldStrategist().advise(
        {
            "round": 2,
            "operator_memory": {
                "counts": {"channel_block": 2},
                "mean_improvement": {"channel_block": 0.1},
            },
            "beam": [
                {
                    "risk": 0.8,
                    "physics_diagnosis": {
                        "hottest_sector": 6,
                        "highest_sensitivity_empty_cells": [{"cell": [2, 3]}],
                    },
                }
            ],
            "recent_outcomes": [{"improvement": 0.1}],
        }
    )
    assert strategy.focus_sector == 6
    assert strategy.operator_weights["channel_block"] > 0.0
    assert strategy.exploration_fraction < 0.3


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


def test_high_fidelity_veto_retains_baseline_when_candidates_are_worse():
    agent = SmartShieldAgent(
        _ScaledFluxEvaluator("fast", 0.5),
        high_fidelity_evaluator=_ScaledFluxEvaluator("verification", 2.0),
        proposal_model=PhysicsGuidedProposalModel(seed=7),
        metric_weights={"far_mean": 1.0},
    )
    result = agent.search(
        rounds=1,
        beam_width=1,
        proposals_per_parent=3,
        high_fidelity_top_k=2,
    )
    assert result.best_fast.design.signature != baseline_lattice_design().signature
    assert result.best_design.signature == baseline_lattice_design().signature
    assert result.verification_stages[0].accepted is False
    assert result.best_high_fidelity_risk == pytest.approx(1.0)


def test_later_verification_stage_can_overrule_an_earlier_acceptance():
    agent = SmartShieldAgent(
        _ScaledFluxEvaluator("fast", 0.5),
        high_fidelity_evaluator=_ScaledFluxEvaluator("cfm", 0.6),
        verification_evaluators=(_ScaledFluxEvaluator("exact", 1.5),),
        proposal_model=PhysicsGuidedProposalModel(seed=9),
        metric_weights={"far_mean": 1.0},
    )
    result = agent.search(
        rounds=1,
        beam_width=1,
        proposals_per_parent=2,
        high_fidelity_top_k=1,
        verification_top_ks=(1, 1),
    )
    assert result.verification_stages[0].accepted is True
    assert result.verification_stages[1].accepted is False
    assert result.best_design.signature == baseline_lattice_design().signature


def test_small_diffusion_search_keeps_all_design_constraints():
    agent = SmartShieldAgent(
        DiffusionShieldEvaluator(14, 14, absorber_uncertainty=(0.0,)),
        proposal_model=PhysicsGuidedProposalModel(seed=11),
    )
    result = agent.search(
        rounds=1,
        beam_width=2,
        proposals_per_parent=4,
        high_fidelity_top_k=0,
    )
    for record in result.records:
        assert record.proposal.design.absorber_count == 11
        assert record.proposal.design.source_cell not in record.proposal.design.absorbers
