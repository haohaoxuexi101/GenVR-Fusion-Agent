from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING
import json
import math
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from .shield_design import (
    ShieldDesign,
    baseline_lattice_design,
    materialize_lattice_design,
    source_distant_mask,
)

if TYPE_CHECKING:
    from .shield_agent_tools import ShieldToolEnvironment


TOOL_COLORS = {
    "generate_structures": "#4c78a8",
    "run_diffusion_batch": "#4c78a8",
    "screen_gmc": "#f58518",
    "promote_genvr": "#f58518",
    "audit_projected_mc": "#54a24b",
    "promote_exact": "#54a24b",
    "certify_mc": "#b279a2",
    "run_unbiased_mc": "#b279a2",
    "finish": "#e45756",
}

FIDELITY_LABELS = {
    "diffusion": "Diffusion",
    "genvr": "GenVR/CFM projected response",
    "exact": "Projected local-MC response",
}

TOOL_DISPLAY_LABELS = {
    "generate_structures": "generate\nstructures",
    "run_diffusion_batch": "diffusion\nbatch",
    "screen_gmc": "GMC\nscreen",
    "promote_genvr": "GenVR/CFM\nresponse",
    "audit_projected_mc": "projected-MC\naudit",
    "promote_exact": "projected local-MC\ndiagnostic",
    "certify_mc": "unbiased MC\ncertification",
    "run_unbiased_mc": "unbiased\nMC",
    "finish": "finish",
}

RESPONSE_FIELD_NAMES = (
    "baseline_genvr_cfm_response",
    "candidate_genvr_cfm_response",
    "baseline_projected_local_mc_response",
    "candidate_projected_local_mc_response",
)


def _response_discretization_from_evaluator(evaluator: Any) -> dict[str, int]:
    discretization = getattr(evaluator, "discretization", None)
    if discretization is None:
        return {}
    values = {}
    for name in ("n_pos", "n_mu", "n_phi", "n_angle", "n_state"):
        value = getattr(discretization, name, None)
        if isinstance(value, (int, np.integer)):
            values[name] = int(value)
    return values


def _response_discretization_from_config(output_dir: Path) -> dict[str, int]:
    config = _read_json(output_dir / "config.json")
    if config is None:
        return {}
    for key in ("response_library", "exact_response_library"):
        path = config.get(key)
        if not isinstance(path, str):
            continue
        match = re.search(r"_p(\d+)_m(\d+)_f(\d+)(?:_|\.)", Path(path).name)
        if match is None:
            continue
        n_pos, n_mu, n_phi = (int(value) for value in match.groups())
        return {"n_pos": n_pos, "n_mu": n_mu, "n_phi": n_phi}
    return {}


def _response_discretization_label(metadata: dict[str, Any] | None) -> str:
    if not isinstance(metadata, dict):
        return ""
    discretization = metadata.get("discretization", metadata)
    if not isinstance(discretization, dict):
        return ""
    values = [discretization.get(name) for name in ("n_pos", "n_mu", "n_phi")]
    if not all(isinstance(value, (int, float)) for value in values):
        return ""
    return f"n_pos={int(values[0])}, n_mu={int(values[1])}, n_phi={int(values[2])}"


def _runtime_response_fields(
    environment: ShieldToolEnvironment | None,
    signature: str | None,
) -> dict[str, Any] | None:
    if environment is None or signature is None:
        return None
    evidence = environment.candidate(signature)
    if evidence is None:
        return None
    arrays: dict[str, Any] = {}
    if evidence.genvr is not None and environment.genvr_baseline is not None:
        arrays["baseline_genvr_cfm_response"] = environment.genvr_baseline.flux
        arrays["candidate_genvr_cfm_response"] = evidence.genvr.flux
    if evidence.exact is not None and environment.exact_baseline is not None:
        arrays["baseline_projected_local_mc_response"] = environment.exact_baseline.flux
        arrays["candidate_projected_local_mc_response"] = evidence.exact.flux
    if not arrays:
        return None
    discretization = _response_discretization_from_evaluator(environment.genvr_evaluator)
    if not discretization:
        discretization = _response_discretization_from_evaluator(environment.exact_evaluator)
    arrays["metadata"] = {
        "schema_version": "1.0",
        "candidate_signature": signature,
        "discretization": discretization,
        "genvr_method": "GenVR/CFM projected response",
        "diagnostic_method": "Projected local-MC response",
        "diagnostic_scope": (
            "Both fields use the same finite interface projection; the local-MC field "
            "is a kernel diagnostic, not an exact continuum truth."
        ),
    }
    return arrays


def _write_response_fields(
    response_fields: dict[str, Any] | None,
    output_dir: Path,
) -> None:
    if not isinstance(response_fields, dict):
        return
    arrays = {
        name: np.asarray(response_fields[name], dtype=np.float64)
        for name in RESPONSE_FIELD_NAMES
        if name in response_fields
    }
    if not arrays:
        return
    np.savez_compressed(output_dir / "response_fields.npz", **arrays)
    metadata = response_fields.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    (output_dir / "response_fields.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_response_fields(output_dir: Path) -> dict[str, Any] | None:
    fields_path = output_dir / "response_fields.npz"
    if not fields_path.exists():
        return None
    with np.load(fields_path) as archive:
        fields = {
            name: np.asarray(archive[name], dtype=np.float64)
            for name in RESPONSE_FIELD_NAMES
            if name in archive.files
        }
    if not fields:
        return None
    metadata = _read_json(output_dir / "response_fields.json") or {}
    if not metadata.get("discretization"):
        metadata["discretization"] = _response_discretization_from_config(output_dir)
    fields["metadata"] = metadata
    return fields


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _read_trajectory(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            events.append(value)
    return events


def _candidate_payloads(summary: dict[str, Any]) -> list[dict[str, Any]]:
    values = summary.get("candidate_archive", [])
    return [value for value in values if isinstance(value, dict)]


def _baseline_payload(summary: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (value for value in _candidate_payloads(summary) if value.get("is_baseline")),
        None,
    )


def _stage_risk(payload: dict[str, Any], stage: str) -> float | None:
    value = payload.get(stage)
    if not isinstance(value, dict):
        return None
    risk = value.get("risk")
    if not isinstance(risk, (int, float)) or not math.isfinite(float(risk)):
        return None
    return float(risk)


def _selected_payload(summary: dict[str, Any]) -> dict[str, Any] | None:
    payloads = [value for value in _candidate_payloads(summary) if not value.get("is_baseline")]
    requested = summary.get("candidate_signature")
    if requested is not None:
        selected = next((value for value in payloads if value.get("signature") == requested), None)
        if selected is not None:
            return selected
    fidelity_rank = {
        "unbiased-mc": 4,
        "projected-local-mc-response": 3,
        "exact-local-mc-response": 3,
        "genvr-cfm-response": 2,
        "diffusion-ensemble": 1,
        "unevaluated": 0,
    }

    def key(value: dict[str, Any]) -> tuple[int, float]:
        rank = fidelity_rank.get(str(value.get("highest_fidelity")), 0)
        risk = next(
            (
                candidate
                for candidate in (
                    _stage_risk(value, "exact"),
                    _stage_risk(value, "genvr"),
                    _stage_risk(value, "diffusion"),
                )
                if candidate is not None
            ),
            math.inf,
        )
        return (-rank, risk)

    return min(payloads, key=key) if payloads else None


def _design_from_payload(
    payload: dict[str, Any] | None,
    baseline: ShieldDesign,
) -> ShieldDesign:
    if payload is None or payload.get("is_baseline"):
        return baseline
    final_design = payload.get("design")
    if isinstance(final_design, dict) and isinstance(final_design.get("absorbers"), list):
        return ShieldDesign(
            tuple(tuple(int(item) for item in cell) for cell in final_design["absorbers"]),
            int(final_design.get("grid_size", baseline.grid_size)),
            tuple(int(item) for item in final_design.get("source_cell", baseline.source_cell)),
        )
    cells = set(baseline.absorbers)
    for cell in payload.get("removed_from_baseline", []):
        cells.discard(tuple(int(item) for item in cell))
    for cell in payload.get("added_to_baseline", []):
        cells.add(tuple(int(item) for item in cell))
    return ShieldDesign(tuple(cells), baseline.grid_size, baseline.source_cell)


def _artifact_path(raw: Any, output_dir: Path) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if path.exists():
        return path
    candidate = output_dir / path
    return candidate if candidate.exists() else None


def _mc_artifacts(
    payload: dict[str, Any] | None,
    output_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, np.ndarray] | None]:
    if payload is None:
        return None, None
    latest = payload.get("latest_mc")
    artifact = latest.get("artifact", {}) if isinstance(latest, dict) else {}
    summary_path = _artifact_path(artifact.get("summary"), output_dir)
    fields_path = _artifact_path(artifact.get("fields"), output_dir)
    signature = str(payload.get("signature", ""))
    if summary_path is None:
        matches = sorted(output_dir.glob(f"mc/*_{signature}/certification.json"))
        summary_path = matches[-1] if matches else None
    if fields_path is None:
        matches = sorted(output_dir.glob(f"mc/*_{signature}/fields.npz"))
        fields_path = matches[-1] if matches else None
    mc_summary = _read_json(summary_path) if summary_path is not None else None
    if fields_path is None:
        return mc_summary, None
    with np.load(fields_path) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return mc_summary, arrays


def _accepted_actions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions = []
    for event in events:
        if event.get("event_type") != "agent_output" or event.get("accepted") is not True:
            continue
        action = event.get("action")
        if not isinstance(action, dict):
            continue
        actions.append({"step": int(event.get("step", len(actions) + 1)), **action})
    return actions


def _baseline_metrics_by_stage(
    summary: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    baseline = _baseline_payload(summary)
    if baseline is not None:
        for stage in ("diffusion", "genvr", "exact"):
            evaluation = baseline.get(stage)
            if isinstance(evaluation, dict) and isinstance(
                evaluation.get("metrics"),
                dict,
            ):
                result[stage] = {
                    str(name): float(value)
                    for name, value in evaluation["metrics"].items()
                    if isinstance(value, (int, float))
                }
    for event in events:
        if event.get("event_type") != "tool_result":
            continue
        payload = event.get("result")
        if not isinstance(payload, dict):
            continue
        stage = payload.get("stage")
        metrics = payload.get("baseline_metrics")
        stage_alias = {
            "gmc_screening": "genvr",
            "projected_mc_audit": "exact",
            "genvr": "genvr",
            "exact": "exact",
        }.get(str(stage))
        if stage_alias is not None and isinstance(metrics, dict):
            result[stage_alias] = {
                str(name): float(value)
                for name, value in metrics.items()
                if isinstance(value, (int, float))
            }
    return result


def _plot_layout(axis, design: ShieldDesign, title: str) -> None:
    image = axis.imshow(
        design.mask(),
        origin="lower",
        extent=[0.0, design.grid_size, 0.0, design.grid_size],
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    source_row, source_column = design.source_cell
    axis.add_patch(
        Rectangle(
            (source_column, source_row),
            1.0,
            1.0,
            fill=False,
            edgecolor="cyan",
            linewidth=2.0,
            linestyle="--",
        )
    )
    axis.set_title(title)
    axis.set_xlabel("x [cm]")
    axis.set_ylabel("y [cm]")
    axis.set_xlim(0.0, design.grid_size)
    axis.set_ylim(0.0, design.grid_size)
    return image


def _overlay_regions(
    axis,
    far_mask: np.ndarray | None,
    deep_mask: np.ndarray | None,
    extent: list[float],
) -> None:
    if far_mask is not None and np.any(far_mask):
        axis.contour(
            far_mask,
            levels=[0.5],
            origin="lower",
            extent=extent,
            colors="lime",
            linewidths=1.1,
        )
    if deep_mask is not None and np.any(deep_mask):
        axis.contour(
            deep_mask,
            levels=[0.5],
            origin="lower",
            extent=extent,
            colors="cyan",
            linewidths=0.9,
            linestyles="--",
        )


def _normalized_log_pair(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    scale = max(float(np.max(left)), float(np.max(right)), 1.0e-300)
    left_normalized = np.asarray(left, dtype=np.float64) / scale
    right_normalized = np.asarray(right, dtype=np.float64) / scale
    positive = np.concatenate(
        [
            left_normalized[left_normalized > 0.0],
            right_normalized[right_normalized > 0.0],
        ]
    )
    floor = max(float(np.quantile(positive, 0.005)), 1.0e-12)
    return (
        np.log10(np.maximum(left_normalized, floor)),
        np.log10(np.maximum(right_normalized, floor)),
        math.log10(floor),
    )


def _plot_search_dashboard(
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    gmc_only = summary.get("discovery_mode") == "gmc_only"
    selected = _selected_payload(summary)
    selected_signature = selected.get("signature") if selected is not None else None
    actions = _accepted_actions(events)
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True)

    tool_order = list(TOOL_COLORS)
    tool_y = {name: index for index, name in enumerate(tool_order)}
    for action in actions:
        tool = str(action.get("tool"))
        step = int(action.get("step", 0))
        axes[0, 0].scatter(
            step,
            tool_y.get(tool, -1),
            s=120,
            color=TOOL_COLORS.get(tool, "gray"),
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        axes[0, 0].text(step, tool_y.get(tool, -1) + 0.18, str(step), ha="center", fontsize=8)
    rejected_steps = [
        int(event.get("step", 0))
        for event in events
        if event.get("event_type") == "agent_output" and event.get("accepted") is False
    ]
    if rejected_steps:
        axes[0, 0].scatter(
            rejected_steps,
            [-0.65] * len(rejected_steps),
            marker="x",
            s=70,
            color="red",
            label="invalid response",
        )
    axes[0, 0].set_yticks(
        range(len(tool_order)),
        [TOOL_DISPLAY_LABELS.get(name, name.replace("_", "\n")) for name in tool_order],
    )
    axes[0, 0].set_xlabel("Agent decision step")
    axes[0, 0].set_title("LLM scientific-tool sequence")
    axes[0, 0].grid(axis="x", alpha=0.25)
    if rejected_steps:
        axes[0, 0].legend(fontsize=8)

    records = [value for value in summary.get("search_records", []) if isinstance(value, dict)]
    risk_key = "gmc_risk" if gmc_only else "diffusion_risk"
    screened_records = [
        value for value in records if value.get(risk_key) is not None
    ]
    risks = np.asarray(
        [float(value[risk_key]) for value in screened_records],
        dtype=np.float64,
    )
    if risks.size:
        x_values = np.arange(1, risks.size + 1)
        operators = [
            str(value.get("mutation", {}).get("operator", "unknown"))
            for value in screened_records
        ]
        operator_names = list(dict.fromkeys(operators))
        color_map = plt.get_cmap("tab10")
        operator_colors = {
            name: color_map(index % 10) for index, name in enumerate(operator_names)
        }
        for index, (risk, operator) in enumerate(zip(risks, operators, strict=False), start=1):
            axes[0, 1].scatter(index, risk, color=operator_colors[operator], s=35, alpha=0.8)
        axes[0, 1].plot(x_values, np.minimum.accumulate(risks), color="black", linewidth=2.0, label="best so far")
        axes[0, 1].axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="baseline")
        for name, color in operator_colors.items():
            axes[0, 1].scatter([], [], color=color, label=name)
        axes[0, 1].set_yscale("log")
        axes[0, 1].legend(fontsize=7, ncol=2)
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No GMC-screened candidates" if gmc_only else "No diffusion search records",
            ha="center",
            va="center",
        )
    axes[0, 1].set_xlabel(
        "GMC screening order" if gmc_only else "Evaluated legal design"
    )
    axes[0, 1].set_ylabel("Normalized GMC risk" if gmc_only else "Robust normalized risk")
    axes[0, 1].set_title(
        "GMC discovery and best-so-far improvement"
        if gmc_only
        else "Diffusion search and best-so-far improvement"
    )
    axes[0, 1].grid(alpha=0.2)

    candidates = [
        value
        for value in _candidate_payloads(summary)
        if not value.get("is_baseline")
        and (_stage_risk(value, "genvr") is not None or _stage_risk(value, "exact") is not None)
    ]
    candidates.sort(
        key=lambda value: (
            _stage_risk(value, "genvr")
            if gmc_only
            else _stage_risk(value, "diffusion")
        )
        or math.inf
    )
    candidates = candidates[:8]
    for candidate in candidates:
        latest_mc = candidate.get("latest_mc")
        mc_far_risk = None
        if isinstance(latest_mc, dict):
            value = (
                latest_mc.get("design_comparison", {})
                .get("far_field", {})
                .get("candidate_over_baseline")
            )
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                mc_far_risk = float(value)
        values = (
            [
                _stage_risk(candidate, "genvr"),
                _stage_risk(candidate, "exact"),
                mc_far_risk,
            ]
            if gmc_only
            else [
                _stage_risk(candidate, "diffusion"),
                _stage_risk(candidate, "genvr"),
                _stage_risk(candidate, "exact"),
                mc_far_risk,
            ]
        )
        points = [(index, value) for index, value in enumerate(values) if value is not None]
        if not points:
            continue
        is_selected = candidate.get("signature") == selected_signature
        axes[0, 2].plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            linewidth=2.6 if is_selected else 1.0,
            alpha=1.0 if is_selected else 0.55,
            color="#d62728" if is_selected else "#4c78a8",
            label=(f"selected {str(selected_signature)[-6:]}" if is_selected else str(candidate.get("signature"))[-6:]),
        )
    axes[0, 2].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    fidelity_labels = (
        ["GMC discovery", "Projected audit", "MC far"]
        if gmc_only
        else ["Diffusion", "GMC", "Projected audit", "MC far"]
    )
    axes[0, 2].set_xticks(range(len(fidelity_labels)), fidelity_labels)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_ylabel("Candidate / baseline risk")
    axes[0, 2].set_title(
        "GMC selection to unbiased-MC certification"
        if gmc_only
        else "Cross-fidelity survival and model disagreement"
    )
    if candidates:
        axes[0, 2].legend(fontsize=7, ncol=2)
    axes[0, 2].grid(alpha=0.2)

    budget = summary.get("budget_usage", {})
    budget_names = list(budget)
    fractions = []
    annotations = []
    for name in budget_names:
        value = budget.get(name, {})
        used = float(value.get("used", 0.0))
        limit = max(float(value.get("limit", 1.0)), 1.0)
        fractions.append(used / limit)
        annotations.append(f"{used:g}/{limit:g}")
    positions = np.arange(len(budget_names))
    axes[1, 0].barh(positions, fractions, color="#72b7b2")
    axes[1, 0].set_yticks(positions, [name.replace("_", " ") for name in budget_names])
    axes[1, 0].set_xlim(0.0, 1.08)
    axes[1, 0].set_xlabel("Fraction of frozen budget used")
    axes[1, 0].set_title("Scientific-compute budget allocation")
    for position, fraction, annotation in zip(positions, fractions, annotations, strict=False):
        axes[1, 0].text(min(fraction + 0.02, 0.99), position, annotation, va="center", fontsize=8)
    axes[1, 0].grid(axis="x", alpha=0.2)

    operator_memory = summary.get("gmc_operator_memory")
    memory_title = "GMC-validated operator memory"
    if (not gmc_only) and (not isinstance(operator_memory, dict) or not any(
        operator_memory.get("counts", {}).values()
    )):
        operator_memory = summary.get("operator_memory", {})
        memory_title = "Diffusion-prefilter operator memory"
    counts = operator_memory.get("counts", {}) if isinstance(operator_memory, dict) else {}
    improvements = operator_memory.get("mean_improvement", {}) if isinstance(operator_memory, dict) else {}
    operator_names = list(counts)
    x_values = np.arange(len(operator_names))
    axes[1, 1].bar(
        x_values,
        [float(counts.get(name, 0.0)) for name in operator_names],
        color="#4c78a8",
        alpha=0.75,
        label="uses",
    )
    twin = axes[1, 1].twinx()
    twin.plot(
        x_values,
        [float(improvements.get(name, 0.0)) for name in operator_names],
        color="#e45756",
        marker="o",
        label="mean improvement",
    )
    axes[1, 1].set_xticks(x_values, [name.replace("_", "\n") for name in operator_names], fontsize=7)
    axes[1, 1].set_ylabel("Accepted proposal count")
    twin.set_ylabel("Mean parent-risk improvement")
    axes[1, 1].set_title(memory_title)
    axes[1, 1].grid(axis="y", alpha=0.2)

    checks = summary.get("verifier_decision", {}).get("checks", {})
    check_items = [
        (name, bool(value.get("passed")))
        for name, value in checks.items()
        if isinstance(value, dict) and isinstance(value.get("passed"), bool)
    ]
    if check_items:
        check_items = check_items[-11:]
        positions = np.arange(len(check_items))
        values = [1.0 if passed else 0.0 for _, passed in check_items]
        colors = ["#54a24b" if passed else "#e45756" for _, passed in check_items]
        axes[1, 2].barh(positions, values, color=colors)
        axes[1, 2].set_yticks(
            positions,
            [name.replace("_", " ") for name, _ in check_items],
            fontsize=8,
        )
        axes[1, 2].set_xlim(0.0, 1.08)
        axes[1, 2].set_xticks([0.0, 1.0], ["fail", "pass"])
    else:
        axes[1, 2].text(0.5, 0.5, "No verifier checks recorded", ha="center", va="center")
    axes[1, 2].set_title(
        f"Independent verifier: {summary.get('final_claim', 'unknown')}"
    )
    axes[1, 2].grid(axis="x", alpha=0.2)

    fig.suptitle(
        (
            "Autonomous GMC-only material-discovery dashboard\n"
            if gmc_only
            else "Autonomous GenShield search dashboard\n"
        )
        +
        f"steps={summary.get('steps_completed', 0)}, selected={selected_signature}, "
        f"termination={summary.get('termination', 'unknown')}",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_gmc_discovery_storyboard(
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    response_fields: dict[str, Any] | None,
    baseline_design: ShieldDesign,
    candidate_design: ShieldDesign,
    output_path: Path,
    dpi: int,
) -> bool:
    if summary.get("discovery_mode") != "gmc_only":
        return False
    if not isinstance(response_fields, dict):
        return False
    baseline_flux = response_fields.get("baseline_genvr_cfm_response")
    candidate_flux = response_fields.get("candidate_genvr_cfm_response")
    if baseline_flux is None or candidate_flux is None:
        return False
    baseline_flux = np.asarray(baseline_flux, dtype=np.float64)
    candidate_flux = np.asarray(candidate_flux, dtype=np.float64)
    if baseline_flux.shape != candidate_flux.shape or baseline_flux.ndim != 2:
        return False

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    actions = _accepted_actions(events)
    action_labels = [
        TOOL_DISPLAY_LABELS.get(str(action.get("tool")), str(action.get("tool")))
        .replace("\n", " ")
        for action in actions
    ]
    action_steps = [int(action.get("step", index + 1)) for index, action in enumerate(actions)]
    action_colors = [
        TOOL_COLORS.get(str(action.get("tool")), "gray") for action in actions
    ]
    if actions:
        axes[0, 0].scatter(
            action_steps,
            np.zeros(len(actions)),
            s=180,
            color=action_colors,
            edgecolor="black",
            linewidth=0.6,
        )
        for step, label in zip(action_steps, action_labels, strict=False):
            axes[0, 0].text(
                step,
                0.08 if step % 2 else -0.08,
                f"{step}. {label}",
                ha="center",
                va="bottom" if step % 2 else "top",
                fontsize=8,
                rotation=20,
            )
        axes[0, 0].plot(action_steps, np.zeros(len(actions)), color="0.45", zorder=0)
    axes[0, 0].set_ylim(-0.32, 0.32)
    axes[0, 0].set_yticks([])
    axes[0, 0].set_xlabel("Autonomous decision step")
    axes[0, 0].set_title("Agent chooses the experiment sequence")
    axes[0, 0].grid(axis="x", alpha=0.2)

    screened = [
        record
        for record in summary.get("search_records", [])
        if isinstance(record, dict) and record.get("gmc_risk") is not None
    ]
    if screened:
        screening_order = np.arange(1, len(screened) + 1)
        risks = np.asarray([float(record["gmc_risk"]) for record in screened])
        generations = np.asarray(
            [int(record.get("generation_round", 0)) for record in screened]
        )
        for generation in sorted(set(generations.tolist())):
            mask = generations == generation
            axes[0, 1].scatter(
                screening_order[mask],
                risks[mask],
                s=70,
                label=f"generation {generation}",
            )
        axes[0, 1].plot(
            screening_order,
            np.minimum.accumulate(risks),
            color="black",
            linewidth=2.2,
            label="best GMC risk",
        )
        axes[0, 1].axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        axes[0, 1].set_yscale("log")
        axes[0, 1].legend(fontsize=8)
    else:
        axes[0, 1].text(0.5, 0.5, "No GMC screening evidence", ha="center", va="center")
    axes[0, 1].set_xlabel("GMC-screened structure")
    axes[0, 1].set_ylabel("Candidate / baseline risk")
    axes[0, 1].set_title("GMC ranks successive material generations")
    axes[0, 1].grid(alpha=0.2)

    memory = summary.get("gmc_operator_memory", {})
    counts = memory.get("counts", {}) if isinstance(memory, dict) else {}
    improvements = memory.get("mean_improvement", {}) if isinstance(memory, dict) else {}
    active = [name for name in counts if int(counts.get(name, 0)) > 0]
    if active:
        positions = np.arange(len(active))
        axes[0, 2].bar(
            positions,
            [float(counts[name]) for name in active],
            color="#4c78a8",
            alpha=0.75,
        )
        twin = axes[0, 2].twinx()
        twin.plot(
            positions,
            [float(improvements.get(name, 0.0)) for name in active],
            color="#e45756",
            marker="o",
            linewidth=2.0,
        )
        axes[0, 2].set_xticks(
            positions,
            [name.replace("_", "\n") for name in active],
            fontsize=8,
        )
        axes[0, 2].set_ylabel("GMC-tested uses")
        twin.set_ylabel("Mean GMC risk improvement")
    else:
        axes[0, 2].text(0.5, 0.5, "No learned operator outcomes", ha="center", va="center")
    axes[0, 2].set_title("The agent learns which structure operators work")
    axes[0, 2].grid(axis="y", alpha=0.2)

    baseline_log, candidate_log, vmin = _normalized_log_pair(
        baseline_flux,
        candidate_flux,
    )
    extent = [0.0, baseline_design.grid_size, 0.0, baseline_design.grid_size]
    flux_image = None
    for axis, field, title in (
        (axes[1, 0], baseline_log, "Baseline high-fidelity GMC flux"),
        (axes[1, 1], candidate_log, "Selected design high-fidelity GMC flux"),
    ):
        flux_image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=vmin,
            vmax=0.0,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xlabel("x [cm]")
        axis.set_ylabel("y [cm]")
    if flux_image is not None:
        fig.colorbar(
            flux_image,
            ax=[axes[1, 0], axes[1, 1]],
            label="log10 flux, common normalization",
            shrink=0.80,
        )

    ratio = np.log10(
        np.maximum(candidate_flux, 1.0e-300)
        / np.maximum(baseline_flux, 1.0e-300)
    )
    ratio_image = axes[1, 2].imshow(
        np.clip(ratio, -1.0, 1.0),
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[1, 2].set_title("GMC candidate / baseline flux")
    axes[1, 2].set_xlabel("x [cm]")
    axes[1, 2].set_ylabel("y [cm]")
    fig.colorbar(ratio_image, ax=axes[1, 2], label="log10 flux ratio")

    selected = _selected_payload(summary)
    mc_ratio = None
    mc_z = None
    if isinstance(selected, dict):
        latest_mc = selected.get("latest_mc")
        comparison = (
            latest_mc.get("design_comparison", {}).get("far_field", {})
            if isinstance(latest_mc, dict)
            else {}
        )
        if isinstance(comparison, dict):
            mc_ratio = comparison.get("candidate_over_baseline")
            mc_z = comparison.get("difference_z")
    progress = summary.get("discovery_progress", {})
    annotation = (
        f"Generated: {progress.get('generated_structures', 0)}\n"
        f"GMC screened: {progress.get('gmc_screened_structures', 0)}\n"
        f"GMC-guided + rescreened: {progress.get('gmc_guided_and_screened', 0)}\n"
        f"MC far ratio: {float(mc_ratio):.3f}\n"
        f"MC difference z: {float(mc_z):.2f}"
        if isinstance(mc_ratio, (int, float)) and isinstance(mc_z, (int, float))
        else (
            f"Generated: {progress.get('generated_structures', 0)}\n"
            f"GMC screened: {progress.get('gmc_screened_structures', 0)}\n"
            f"GMC-guided + rescreened: {progress.get('gmc_guided_and_screened', 0)}"
        )
    )
    axes[1, 2].text(
        0.02,
        0.02,
        annotation,
        transform=axes[1, 2].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.4"},
    )

    discretization = _response_discretization_label(
        response_fields.get("metadata")
    )
    fig.suptitle(
        "Final-presentation workflow: autonomous GMC discovery → GMC weight windows → unbiased MC\n"
        f"{discretization}; selected={candidate_design.signature}; "
        f"verifier={summary.get('final_claim', 'unknown')}",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_decision_timeline(
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> bool:
    actions = _accepted_actions(events)
    if not actions:
        return False
    row_height = 1.65
    fig_height = max(5.0, 1.0 + row_height * len(actions))
    fig, axis = plt.subplots(figsize=(18, fig_height), constrained_layout=True)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, len(actions))
    axis.axis("off")
    for row, action in enumerate(actions):
        y = len(actions) - row - 1
        tool = str(action.get("tool", "unknown"))
        color = TOOL_COLORS.get(tool, "gray")
        axis.add_patch(
            Rectangle((0.01, y + 0.08), 0.98, 0.84, facecolor=color, alpha=0.10, edgecolor=color)
        )
        axis.add_patch(Rectangle((0.01, y + 0.08), 0.012, 0.84, facecolor=color, edgecolor="none"))
        axis.text(
            0.035,
            y + 0.72,
            f"Step {action.get('step')}  |  {TOOL_DISPLAY_LABELS.get(tool, tool).replace(chr(10), ' ')}",
            fontsize=11,
            fontweight="bold",
            color=color,
            va="center",
        )
        hypothesis = textwrap.fill(str(action.get("hypothesis", "")), width=125)
        expected = textwrap.fill(str(action.get("expected_observation", "")), width=125)
        axis.text(0.035, y + 0.48, f"Hypothesis: {hypothesis}", fontsize=8.7, va="center")
        axis.text(0.035, y + 0.22, f"Expected evidence: {expected}", fontsize=8.7, va="center")
    axis.set_title(
        "Agent hypothesis → experiment timeline\n"
        f"final claim: {summary.get('final_claim', 'unknown')}",
        fontsize=16,
        pad=18,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _primary_flux_pair(
    response_fields: dict[str, Any] | None,
    arrays: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    if isinstance(response_fields, dict):
        baseline = response_fields.get("baseline_genvr_cfm_response")
        candidate = response_fields.get("candidate_genvr_cfm_response")
        if baseline is not None and candidate is not None:
            discretization = _response_discretization_label(response_fields.get("metadata"))
            suffix = f" ({discretization})" if discretization else ""
            return (
                np.asarray(baseline, dtype=np.float64),
                np.asarray(candidate, dtype=np.float64),
                f"GenVR/CFM projected response{suffix}",
            )
    if arrays is not None:
        return (
            np.asarray(arrays["baseline_variance_reduced_flux"], dtype=np.float64),
            np.asarray(arrays["candidate_variance_reduced_flux"], dtype=np.float64),
            "unbiased weight-window MC",
        )
    return None


def _mc_region_panel(axis, mc_summary: dict[str, Any] | None) -> None:
    if mc_summary is None:
        axis.text(0.5, 0.5, "No unbiased MC certification", ha="center", va="center")
        axis.set_title("Regional certification")
        return
    regions = ("far_field", "deep_far_field")
    labels = ("Far field", "Deep far field")
    positions = np.arange(len(regions), dtype=np.float64)
    width = 0.34
    for offset, design_name, color in (
        (-0.5 * width, "baseline", "#4c78a8"),
        (0.5 * width, "candidate", "#f58518"),
    ):
        estimates = []
        errors = []
        for region in regions:
            baseline_region = mc_summary["designs"]["baseline"]["variance_reduced"]["regions"][region]
            selected_region = mc_summary["designs"][design_name]["variance_reduced"]["regions"][region]
            scale = max(float(baseline_region["estimate"]), 1.0e-300)
            estimates.append(float(selected_region["estimate"]) / scale)
            errors.append(1.96 * float(selected_region.get("standard_error") or 0.0) / scale)
        axis.bar(
            positions + offset,
            estimates,
            width=width,
            yerr=errors,
            capsize=4,
            color=color,
            label=design_name,
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Flux / baseline regional estimate")
    axis.set_title("Unbiased MC regional estimates, 95% CI")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)


def _plot_candidate_evidence(
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    output_dir: Path,
    output_path: Path,
    environment: ShieldToolEnvironment | None,
    response_fields: dict[str, Any] | None,
    dpi: int,
) -> bool:
    selected = _selected_payload(summary)
    if selected is None:
        return False
    baseline_design = environment.baseline_design if environment is not None else baseline_lattice_design()
    evidence = environment.candidate(str(selected.get("signature"))) if environment is not None else None
    candidate_design = evidence.design if evidence is not None else _design_from_payload(selected, baseline_design)
    mc_summary, arrays = _mc_artifacts(selected, output_dir)
    primary_pair = _primary_flux_pair(response_fields, arrays)
    if primary_pair is not None:
        baseline_flux, candidate_flux, flux_label = primary_pair
        problem = materialize_lattice_design(
            candidate_design,
            candidate_flux.shape[1],
            candidate_flux.shape[0],
        )
        if flux_label == "unbiased weight-window MC" and arrays is not None:
            far_mask = np.asarray(arrays.get("candidate_far_mask"), dtype=bool)
            deep_mask = np.asarray(arrays.get("candidate_deep_far_mask"), dtype=bool)
        else:
            far_mask = source_distant_mask(problem)
            threshold = float(np.quantile(baseline_flux[far_mask], 0.25))
            deep_mask = far_mask & (baseline_flux <= threshold)
    else:
        baseline_flux = candidate_flux = far_mask = deep_mask = None
        flux_label = "No field artifact"

    fig, axes = plt.subplots(3, 3, figsize=(19, 16), constrained_layout=True)
    baseline_image = _plot_layout(axes[0, 0], baseline_design, "Baseline material layout")
    candidate_image = _plot_layout(
        axes[0, 1],
        candidate_design,
        f"Agent candidate {candidate_design.signature}",
    )
    fig.colorbar(baseline_image, ax=[axes[0, 0], axes[0, 1]], label="absorber occupancy", shrink=0.8)
    delta = candidate_design.mask().astype(float) - baseline_design.mask().astype(float)
    delta_image = axes[0, 2].imshow(
        delta,
        origin="lower",
        extent=[0.0, baseline_design.grid_size, 0.0, baseline_design.grid_size],
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[0, 2].set_title("Compiled edit: red added, blue removed")
    axes[0, 2].set_xlabel("x [cm]")
    axes[0, 2].set_ylabel("y [cm]")
    fig.colorbar(delta_image, ax=axes[0, 2], label="material edit")

    if baseline_flux is not None and candidate_flux is not None:
        baseline_log, candidate_log, log_floor = _normalized_log_pair(baseline_flux, candidate_flux)
        extent = [0.0, baseline_design.grid_size, 0.0, baseline_design.grid_size]
        flux_axes = (axes[1, 0], axes[1, 1])
        if flux_label.startswith("GenVR/CFM"):
            short_flux_label = "GenVR/CFM projected response"
        elif flux_label == "unbiased weight-window MC":
            short_flux_label = "unbiased weight-window MC"
        else:
            short_flux_label = flux_label
        flux_image = None
        for axis, values, title in (
            (axes[1, 0], baseline_log, f"Baseline {short_flux_label}"),
            (axes[1, 1], candidate_log, f"Candidate {short_flux_label}"),
        ):
            flux_image = axis.imshow(
                values,
                origin="lower",
                extent=extent,
                cmap="magma",
                vmin=log_floor,
                vmax=0.0,
                interpolation="nearest",
            )
            _overlay_regions(axis, far_mask, deep_mask, extent)
            axis.set_title(title)
            axis.set_xlabel("x [cm]")
            axis.set_ylabel("y [cm]")
        if flux_image is not None:
            fig.colorbar(flux_image, ax=list(flux_axes), label="log10 flux, common normalization", shrink=0.8)
        log_ratio = np.log10(
            np.clip(
                candidate_flux / np.maximum(baseline_flux, 1.0e-300),
                1.0e-2,
                1.0e2,
            )
        )
        ratio_image = axes[1, 2].imshow(
            log_ratio,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        _overlay_regions(axes[1, 2], far_mask, deep_mask, extent)
        axes[1, 2].set_title(f"Candidate / baseline\n{short_flux_label}")
        axes[1, 2].set_xlabel("x [cm]")
        axes[1, 2].set_ylabel("y [cm]")
        fig.colorbar(ratio_image, ax=axes[1, 2], label="log10 flux ratio")
    else:
        for axis in axes[1]:
            axis.text(0.5, 0.5, "No saved field for this run", ha="center", va="center")
            axis.axis("off")

    risk_labels = []
    risk_values = []
    risk_colors = []
    for stage, color in (("diffusion", "#4c78a8"), ("genvr", "#f58518"), ("exact", "#54a24b")):
        value = _stage_risk(selected, stage)
        if value is not None:
            risk_labels.append(FIDELITY_LABELS[stage])
            risk_values.append(value)
            risk_colors.append(color)
    if mc_summary is not None:
        for region, label, color in (
            ("far_field", "MC far", "#b279a2"),
            ("deep_far_field", "MC deep", "#ff9da6"),
        ):
            value = mc_summary.get("design_comparison", {}).get(region, {}).get("candidate_over_baseline")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                risk_labels.append(label)
                risk_values.append(float(value))
                risk_colors.append(color)
    axes[2, 0].bar(risk_labels, risk_values, color=risk_colors)
    axes[2, 0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[2, 0].set_ylabel("Candidate / baseline")
    axes[2, 0].set_title("Evidence ladder across fidelity levels")
    axes[2, 0].tick_params(axis="x", rotation=25)
    axes[2, 0].grid(axis="y", alpha=0.2)

    baseline_metrics = _baseline_metrics_by_stage(summary, events)
    metric_names = ("far_mean", "far_cvar90", "far_max", "worst_sector_mean")
    x_values = np.arange(len(metric_names), dtype=np.float64)
    available_stages = []
    for stage in ("diffusion", "genvr", "exact"):
        candidate_stage = selected.get(stage)
        candidate_metrics = candidate_stage.get("metrics") if isinstance(candidate_stage, dict) else None
        stage_baseline = baseline_metrics.get(stage)
        if not isinstance(candidate_metrics, dict) or not stage_baseline:
            continue
        available_stages.append((stage, candidate_metrics, stage_baseline))
    width = 0.8 / max(len(available_stages), 1)
    for index, (stage, candidate_metrics, stage_baseline) in enumerate(available_stages):
        ratios = [
            float(candidate_metrics[name]) / max(float(stage_baseline[name]), 1.0e-300)
            for name in metric_names
        ]
        offset = (index - 0.5 * (len(available_stages) - 1)) * width
        axes[2, 1].bar(
            x_values + offset,
            ratios,
            width=width,
            label=FIDELITY_LABELS[stage],
        )
    axes[2, 1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[2, 1].set_xticks(x_values, [name.replace("_", "\n") for name in metric_names])
    axes[2, 1].set_ylabel("Metric / same-fidelity baseline")
    axes[2, 1].set_title("Which shielding objectives improved?")
    if available_stages:
        axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(axis="y", alpha=0.2)

    _mc_region_panel(axes[2, 2], mc_summary)
    handles = [
        Line2D([0], [0], color="lime", linewidth=1.2, label="far-field metric"),
        Line2D([0], [0], color="cyan", linewidth=1.0, linestyle="--", label="deep diagnostic"),
    ]
    axes[1, 0].legend(handles=handles, loc="lower left", fontsize=8)
    fig.suptitle(
        "Autonomous GMC-guided material-discovery evidence chain\n"
        f"selected={candidate_design.signature}; primary field={flux_label}; "
        f"final claim={summary.get('final_claim', 'unknown')}",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _gmc_screening_entries(
    summary: dict[str, Any],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = {
        str(payload.get("signature")): payload
        for payload in _candidate_payloads(summary)
        if payload.get("signature") is not None
    }
    records = {
        str(record.get("design_signature")): record
        for record in summary.get("search_records", [])
        if isinstance(record, dict) and record.get("design_signature") is not None
    }
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        if event.get("event_type") != "tool_result":
            continue
        result = event.get("result")
        if not isinstance(result, dict) or result.get("stage") not in {
            "gmc_screening",
            "genvr",
        }:
            continue
        values = result.get("results")
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, dict) or value.get("signature") is None:
                continue
            signature = str(value["signature"])
            risk = value.get("risk")
            if signature in seen or not isinstance(risk, (int, float)):
                continue
            payload = candidates.get(signature, {})
            record = records.get(signature, {})
            lineage = payload.get("lineage", {}) if isinstance(payload, dict) else {}
            mutation = value.get("mutation")
            if not isinstance(mutation, dict):
                mutation = lineage.get("mutation") if isinstance(lineage, dict) else None
            if not isinstance(mutation, dict):
                mutation = record.get("mutation") if isinstance(record, dict) else None
            generation = value.get("generation_round")
            if not isinstance(generation, (int, float)) and isinstance(lineage, dict):
                generation = lineage.get("generation_round")
            if not isinstance(generation, (int, float)) and isinstance(record, dict):
                generation = record.get("generation_round")
            guided = value.get("gmc_guided_generation")
            if not isinstance(guided, bool) and isinstance(lineage, dict):
                guided = lineage.get("gmc_guided_generation")
            if not isinstance(guided, bool) and isinstance(record, dict):
                guided = record.get("gmc_guided_generation")
            entries.append(
                {
                    "signature": signature,
                    "risk": float(risk),
                    "accepted": bool(value.get("accepted", float(risk) < 1.0)),
                    "runtime_s": float(value.get("runtime_s", 0.0)),
                    "generation_round": int(generation or 0),
                    "gmc_guided_generation": bool(guided),
                    "mutation": mutation or {},
                    "payload": payload,
                    "step": int(event.get("step", len(entries) + 1)),
                }
            )
            seen.add(signature)
    if entries:
        return entries

    for record in summary.get("search_records", []):
        if not isinstance(record, dict) or record.get("design_signature") is None:
            continue
        risk = record.get("gmc_risk")
        if not isinstance(risk, (int, float)):
            continue
        signature = str(record["design_signature"])
        if signature in seen:
            continue
        payload = candidates.get(signature, {})
        entries.append(
            {
                "signature": signature,
                "risk": float(risk),
                "accepted": bool(record.get("gmc_accepted", float(risk) < 1.0)),
                "runtime_s": float(record.get("gmc_runtime_s", 0.0)),
                "generation_round": int(record.get("generation_round", 0)),
                "gmc_guided_generation": bool(
                    record.get("gmc_guided_generation", False)
                ),
                "mutation": record.get("mutation", {}),
                "payload": payload,
                "step": len(entries) + 1,
            }
        )
        seen.add(signature)
    if entries:
        return entries

    for payload in candidates.values():
        if payload.get("is_baseline"):
            continue
        risk = _stage_risk(payload, "gmc")
        if risk is None:
            risk = _stage_risk(payload, "genvr")
        if risk is None:
            continue
        lineage = payload.get("lineage", {})
        evaluation = payload.get("gmc") or payload.get("genvr") or {}
        entries.append(
            {
                "signature": str(payload.get("signature")),
                "risk": risk,
                "accepted": risk < 1.0,
                "runtime_s": float(evaluation.get("runtime_s", 0.0)),
                "generation_round": int(lineage.get("generation_round", 0)),
                "gmc_guided_generation": bool(
                    lineage.get("gmc_guided_generation", False)
                ),
                "mutation": lineage.get("mutation") or {},
                "payload": payload,
                "step": len(entries) + 1,
            }
        )
    return entries


def _animation_design(
    entry: dict[str, Any] | None,
    baseline_design: ShieldDesign,
) -> ShieldDesign:
    payload = entry.get("payload") if isinstance(entry, dict) else None
    return _design_from_payload(payload if isinstance(payload, dict) else None, baseline_design)


def _render_gmc_screening_frame(
    summary: dict[str, Any],
    entries: list[dict[str, Any]],
    baseline_design: ShieldDesign,
    frame: dict[str, Any],
    dpi: int,
) -> Any:
    from PIL import Image

    generation_colors = {
        0: "#9d9da1",
        1: "#4c78a8",
        2: "#f58518",
        3: "#54a24b",
        4: "#b279a2",
    }
    phase = str(frame["phase"])
    evaluated_count = int(frame.get("evaluated_count", 0))
    current = frame.get("entry")
    evaluated = entries[:evaluated_count]
    selected_signature = str(summary.get("candidate_signature") or "")
    if phase == "intro":
        design = baseline_design
    elif isinstance(current, dict):
        design = _animation_design(current, baseline_design)
    else:
        selected = next(
            (entry for entry in entries if entry["signature"] == selected_signature),
            min(entries, key=lambda entry: float(entry["risk"])),
        )
        current = selected
        design = _animation_design(selected, baseline_design)

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#f5f7fa", constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.03, 1.28, 1.02),
        height_ratios=(1.0, 1.0),
    )
    layout_axis = fig.add_subplot(grid[:, 0])
    trace_axis = fig.add_subplot(grid[0, 1:])
    board_axis = fig.add_subplot(grid[1, 1])
    decision_axis = fig.add_subplot(grid[1, 2])

    layout_axis.imshow(
        design.mask(),
        origin="lower",
        extent=[0.0, design.grid_size, 0.0, design.grid_size],
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    source_row, source_column = design.source_cell
    layout_axis.add_patch(
        Rectangle(
            (source_column, source_row),
            1.0,
            1.0,
            fill=False,
            edgecolor="#00bcd4",
            linewidth=2.4,
            linestyle="--",
        )
    )
    payload = current.get("payload", {}) if isinstance(current, dict) else {}
    removed_cells = payload.get("removed_from_baseline", []) if isinstance(payload, dict) else []
    added_cells = payload.get("added_to_baseline", []) if isinstance(payload, dict) else []
    for cell in removed_cells:
        row, column = (int(value) for value in cell)
        layout_axis.add_patch(
            Rectangle(
                (column, row),
                1.0,
                1.0,
                fill=False,
                edgecolor="#e45756",
                linewidth=2.8,
                linestyle="--",
            )
        )
    for cell in added_cells:
        row, column = (int(value) for value in cell)
        layout_axis.add_patch(
            Rectangle(
                (column, row),
                1.0,
                1.0,
                fill=False,
                edgecolor="#2ca02c",
                linewidth=3.0,
            )
        )
    layout_axis.set_xticks(np.arange(design.grid_size + 1), minor=True)
    layout_axis.set_yticks(np.arange(design.grid_size + 1), minor=True)
    layout_axis.grid(which="minor", color="white", linewidth=0.65, alpha=0.75)
    layout_axis.tick_params(which="minor", bottom=False, left=False)
    layout_axis.set_xlabel("x macro-cell")
    layout_axis.set_ylabel("y macro-cell")
    if phase == "intro":
        layout_axis.set_title("Frozen baseline layout\n11 absorbers, fixed source and mass", fontsize=12)
    elif phase == "transition" and isinstance(current, dict):
        layout_axis.set_title(
            f"GMC-selected parent for generation {int(frame.get('generation', 1))}\n"
            f"{str(current.get('signature'))[:12]}",
            fontsize=11.5,
        )
    elif phase == "final" and isinstance(current, dict):
        layout_axis.set_title(
            f"Selected GMC winner\n{str(current.get('signature'))[:12]}",
            fontsize=12,
        )
    elif isinstance(current, dict):
        guided = "GMC-guided · " if current.get("gmc_guided_generation") else ""
        layout_axis.set_title(
            f"Candidate {evaluated_count}/{len(entries)} · generation "
            f"{int(current.get('generation_round', 0))}\n"
            f"{guided}{str(current.get('signature'))[:12]}",
            fontsize=12,
        )
    else:
        layout_axis.set_title("Selected GMC winner", fontsize=12)

    risks = np.asarray([max(float(entry["risk"]), 1.0e-8) for entry in entries])
    x_values = np.arange(1, len(entries) + 1)
    trace_axis.axhspan(
        max(float(np.min(risks)) * 0.7, 1.0e-8),
        1.0,
        color="#54a24b",
        alpha=0.08,
    )
    trace_axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1.2)
    trace_axis.text(
        len(entries) + 0.35,
        1.0,
        "baseline",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    if evaluated:
        evaluated_risks = risks[:evaluated_count]
        generations = np.asarray(
            [int(entry.get("generation_round", 0)) for entry in evaluated]
        )
        for generation in sorted(set(generations.tolist())):
            mask = generations == generation
            color = generation_colors.get(generation, "#72b7b2")
            trace_axis.scatter(
                x_values[:evaluated_count][mask],
                evaluated_risks[mask],
                s=82,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=f"generation {generation}",
                zorder=3,
            )
        trace_axis.plot(
            x_values[:evaluated_count],
            np.minimum.accumulate(evaluated_risks),
            color="#1f1f1f",
            linewidth=2.2,
            marker="o",
            markersize=3,
            label="best so far",
            zorder=2,
        )
        if phase == "candidate":
            trace_axis.scatter(
                [evaluated_count],
                [evaluated_risks[-1]],
                s=230,
                facecolors="none",
                edgecolors="#e45756",
                linewidth=2.5,
                zorder=4,
            )
        trace_axis.legend(loc="upper right", fontsize=8, ncol=3)
    trace_axis.set_yscale("log")
    trace_axis.set_xlim(0.5, len(entries) + 0.5)
    trace_axis.set_ylim(
        max(float(np.min(risks)) * 0.7, 1.0e-8),
        max(1.35, float(np.max(risks)) * 1.3),
    )
    trace_axis.set_xticks(x_values)
    trace_axis.set_xlabel("GMC evaluation order")
    trace_axis.set_ylabel("candidate / baseline risk")
    trace_axis.set_title("Live GMC screening trace · lower is better")
    trace_axis.grid(alpha=0.22, which="both")

    if evaluated:
        leaders = sorted(evaluated, key=lambda entry: float(entry["risk"]))[:6]
        positions = np.arange(len(leaders))
        leader_risks = np.asarray([max(float(entry["risk"]), 1.0e-8) for entry in leaders])
        minimum = max(float(np.min(leader_risks)) * 0.75, 1.0e-8)
        colors = [
            "#f2c14e"
            if entry["signature"] == leaders[0]["signature"]
            else generation_colors.get(int(entry.get("generation_round", 0)), "#72b7b2")
            for entry in leaders
        ]
        board_axis.hlines(positions, minimum, leader_risks, color=colors, linewidth=5)
        board_axis.scatter(leader_risks, positions, color=colors, s=90, edgecolor="black")
        for position, entry in zip(positions, leaders, strict=False):
            board_axis.text(
                float(entry["risk"]) * 1.05,
                position,
                f"{float(entry['risk']):.4f}",
                va="center",
                fontsize=8,
            )
        board_axis.set_yticks(
            positions,
            [
                f"#{index + 1} {str(entry['signature'])[:8]} · G{int(entry.get('generation_round', 0))}"
                for index, entry in enumerate(leaders)
            ],
            fontsize=8,
        )
        board_axis.invert_yaxis()
        board_axis.set_xscale("log")
        board_axis.set_xlim(minimum, max(1.25, float(np.max(leader_risks)) * 1.25))
        board_axis.axvline(1.0, color="#555555", linestyle="--", linewidth=1.0)
        board_axis.set_xlabel("GMC risk ratio")
        board_axis.grid(axis="x", alpha=0.22, which="both")
    else:
        board_axis.text(
            0.5,
            0.5,
            "Candidates wait for\nmeasured GMC evidence",
            ha="center",
            va="center",
            fontsize=15,
            transform=board_axis.transAxes,
        )
        board_axis.set_xticks([])
        board_axis.set_yticks([])
    board_axis.set_title("Live tournament leaderboard")

    decision_axis.axis("off")
    usage = summary.get("budget_usage", {})
    gmc_budget = usage.get("genvr_candidates", {}) if isinstance(usage, dict) else {}
    target = max(
        len(entries),
        int(gmc_budget.get("limit", 0)) if isinstance(gmc_budget, dict) else 0,
    )
    target = max(target, 1)
    progress = min(evaluated_count / target, 1.0)
    decision_axis.add_patch(
        Rectangle(
            (0.05, 0.89),
            0.90,
            0.055,
            transform=decision_axis.transAxes,
            facecolor="#d9dde3",
            edgecolor="none",
        )
    )
    decision_axis.add_patch(
        Rectangle(
            (0.05, 0.89),
            0.90 * progress,
            0.055,
            transform=decision_axis.transAxes,
            facecolor="#f58518",
            edgecolor="none",
        )
    )
    decision_axis.text(
        0.05,
        0.965,
        f"GMC budget: {evaluated_count}/{target}",
        transform=decision_axis.transAxes,
        fontsize=11,
        weight="bold",
        va="top",
    )

    if phase == "intro":
        headline = "Agent opens the tournament"
        detail = (
            "Geometry and material inventory are frozen.\n"
            "The Agent decides which legal structures\n"
            "deserve high-fidelity GMC evaluation."
        )
    elif phase == "transition":
        generation = int(frame.get("generation", 1))
        leader = min(evaluated, key=lambda entry: float(entry["risk"]))
        headline = f"GMC evidence creates Generation {generation}"
        detail = (
            f"Best parent: {leader['signature'][:12]}\n"
            f"Measured risk: {float(leader['risk']):.4f}\n"
            "The flux-informed operator now proposes\n"
            "a new generation around the measured path."
        )
    elif phase == "final":
        winner = current
        latest_mc = winner.get("payload", {}).get("latest_mc") if isinstance(winner, dict) else None
        comparison = (
            latest_mc.get("design_comparison", {}).get("far_field", {})
            if isinstance(latest_mc, dict)
            else {}
        )
        ratio = comparison.get("candidate_over_baseline") if isinstance(comparison, dict) else None
        z_score = comparison.get("difference_z") if isinstance(comparison, dict) else None
        headline = "GMC winner advances to unbiased MC"
        detail = (
            f"Selected: {winner['signature'][:12]}\n"
            f"GMC risk: {float(winner['risk']):.4f}\n"
            "Its GMC flux becomes the MC weight window."
        )
        if isinstance(ratio, (int, float)) and isinstance(z_score, (int, float)):
            detail += f"\nMC ratio: {float(ratio):.3f} · z={float(z_score):.2f}"
    else:
        mutation = current.get("mutation", {}) if isinstance(current, dict) else {}
        operator = str(mutation.get("operator", "constraint-safe proposal"))
        removed = mutation.get("remove")
        added = mutation.get("add")
        decision = "accepted" if current.get("accepted") else "rejected"
        headline = f"GMC screen: {decision}"
        detail = (
            f"Operator: {operator}\n"
            f"Swap: {removed} → {added}\n"
            f"Risk: {float(current['risk']):.4f}  "
            f"({100.0 * (1.0 - float(current['risk'])):+.1f}% vs baseline)\n"
            f"Measured runtime: {float(current.get('runtime_s', 0.0)):.1f} s"
        )
    decision_axis.text(
        0.05,
        0.80,
        headline,
        transform=decision_axis.transAxes,
        fontsize=14,
        weight="bold",
        va="top",
        color="#1f3552",
    )
    decision_axis.text(
        0.05,
        0.70,
        detail,
        transform=decision_axis.transAxes,
        fontsize=10.5,
        va="top",
        linespacing=1.45,
    )

    pipeline = ("Generate", "GMC screen", "Learn", "Regenerate", "MC certify")
    if phase == "intro":
        active_stage = 0
    elif phase == "transition":
        active_stage = 2
    elif phase == "final":
        active_stage = 4
    else:
        active_stage = 1 if int(current.get("generation_round", 0)) <= 1 else 3
    pipeline_top = 0.42 if phase == "intro" else 0.35
    pipeline_spacing = 0.077 if phase == "intro" else 0.072
    for index, label in enumerate(pipeline):
        y = pipeline_top - pipeline_spacing * index
        completed = index < active_stage
        active = index == active_stage
        color = "#54a24b" if completed else "#f58518" if active else "#d9dde3"
        text_color = "white" if completed or active else "#555555"
        decision_axis.text(
            0.08,
            y,
            f"{index + 1}  {label}",
            transform=decision_axis.transAxes,
            fontsize=10,
            color=text_color,
            weight="bold" if active else "normal",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": color,
                "edgecolor": "none",
            },
        )

    if phase == "intro":
        title = "GMC FAST-SCREENING TOURNAMENT"
        subtitle = "Autonomous candidate selection begins"
    elif phase == "transition":
        title = "GMC FEEDBACK CLOSES THE DESIGN LOOP"
        subtitle = "Measured transport evidence changes the next generation"
    elif phase == "final":
        title = "GMC SELECTS · MONTE CARLO CERTIFIES"
        subtitle = "The fast model allocates expensive physics only to the winner"
    else:
        title = "GMC FAST-SCREENING TOURNAMENT"
        subtitle = "Candidates are ranked by measured transport risk, not by an LLM guess"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=16, weight="bold")
    scope = (
        "GMC-only campaign"
        if summary.get("discovery_mode") == "gmc_only"
        else "Pilot replay · GMC stages only"
    )
    fig.text(
        0.99,
        0.012,
        f"{scope} · risk < 1 is better · saved evidence",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    image = Image.fromarray(pixels, mode="RGB")
    palette_namespace = getattr(Image, "Palette", None)
    adaptive = palette_namespace.ADAPTIVE if palette_namespace is not None else Image.ADAPTIVE
    return image.convert("P", palette=adaptive, colors=192)


def _plot_gmc_screening_animation(
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    baseline_design: ShieldDesign,
    output_path: Path,
    dpi: int,
) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False
    entries = _gmc_screening_entries(summary, events)
    if not entries:
        return False
    frames: list[dict[str, Any]] = [{"phase": "intro", "evaluated_count": 0}]
    previous_generation = None
    for index, entry in enumerate(entries, start=1):
        generation = int(entry.get("generation_round", 0))
        if previous_generation is not None and generation != previous_generation:
            frames.append(
                {
                    "phase": "transition",
                    "evaluated_count": index - 1,
                    "generation": generation,
                    "entry": entries[index - 2],
                }
            )
        frames.append(
            {
                "phase": "candidate",
                "evaluated_count": index,
                "entry": entry,
            }
        )
        previous_generation = generation
    selected_signature = str(summary.get("candidate_signature") or "")
    selected = next(
        (entry for entry in entries if entry["signature"] == selected_signature),
        min(entries, key=lambda entry: float(entry["risk"])),
    )
    frames.append(
        {
            "phase": "final",
            "evaluated_count": len(entries),
            "entry": selected,
        }
    )
    animation_dpi = min(max(int(dpi), 60), 110)
    images = [
        _render_gmc_screening_frame(
            summary,
            entries,
            baseline_design,
            frame,
            animation_dpi,
        )
        for frame in frames
    ]
    durations = [1700]
    durations.extend(
        1500 if frame["phase"] == "transition" else 900
        for frame in frames[1:-1]
    )
    durations.append(2800)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    for image in images:
        image.close()
    return output_path.exists() and output_path.stat().st_size > 0


def _plot_response_comparison(
    response_fields: dict[str, Any] | None,
    baseline_design: ShieldDesign,
    candidate_design: ShieldDesign,
    output_path: Path,
    dpi: int,
) -> bool:
    if not isinstance(response_fields, dict):
        return False
    if any(name not in response_fields for name in RESPONSE_FIELD_NAMES):
        return False
    fields = {
        name: np.asarray(response_fields[name], dtype=np.float64)
        for name in RESPONSE_FIELD_NAMES
    }
    shapes = {values.shape for values in fields.values()}
    if len(shapes) != 1 or any(values.ndim != 2 for values in fields.values()):
        return False

    scale = max(max(float(np.max(values)) for values in fields.values()), 1.0e-300)
    normalized = {name: values / scale for name, values in fields.items()}
    positive = np.concatenate(
        [values[values > 0.0] for values in normalized.values() if np.any(values > 0.0)]
    )
    floor = max(float(np.quantile(positive, 0.005)), 1.0e-12) if positive.size else 1.0e-12
    log_fields = {
        name: np.log10(np.maximum(values, floor))
        for name, values in normalized.items()
    }
    ratios = {
        "genvr": np.log10(
            np.clip(
                fields["candidate_genvr_cfm_response"]
                / np.maximum(fields["baseline_genvr_cfm_response"], 1.0e-300),
                1.0e-2,
                1.0e2,
            )
        ),
        "projected": np.log10(
            np.clip(
                fields["candidate_projected_local_mc_response"]
                / np.maximum(fields["baseline_projected_local_mc_response"], 1.0e-300),
                1.0e-2,
                1.0e2,
            )
        ),
    }
    extent = [0.0, baseline_design.grid_size, 0.0, baseline_design.grid_size]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    flux_image = None
    panels = (
        (axes[0, 0], log_fields["baseline_genvr_cfm_response"], "Baseline GenVR/CFM response"),
        (axes[0, 1], log_fields["candidate_genvr_cfm_response"], "Candidate GenVR/CFM response"),
        (
            axes[1, 0],
            log_fields["baseline_projected_local_mc_response"],
            "Baseline projected local-MC diagnostic",
        ),
        (
            axes[1, 1],
            log_fields["candidate_projected_local_mc_response"],
            "Candidate projected local-MC diagnostic",
        ),
    )
    for axis, values, title in panels:
        flux_image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=math.log10(floor),
            vmax=0.0,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xlabel("x [cm]")
        axis.set_ylabel("y [cm]")
    if flux_image is not None:
        fig.colorbar(
            flux_image,
            ax=[axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
            label="log10 flux, common absolute normalization",
            shrink=0.78,
        )

    ratio_image = None
    for axis, values, title in (
        (axes[0, 2], ratios["genvr"], "GenVR/CFM candidate / baseline"),
        (axes[1, 2], ratios["projected"], "Projected local-MC candidate / baseline"),
    ):
        ratio_image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xlabel("x [cm]")
        axis.set_ylabel("y [cm]")
    if ratio_image is not None:
        fig.colorbar(
            ratio_image,
            ax=[axes[0, 2], axes[1, 2]],
            label="log10 flux ratio (clipped to 0.1–10)",
            shrink=0.78,
        )

    discretization = _response_discretization_label(response_fields.get("metadata"))
    discretization_text = discretization or "finite interface phase space"
    fig.suptitle(
        "Response-method comparison for the selected shield\n"
        f"{discretization_text}; both rows share the same projection. "
        "Projected local-MC is a diagnostic, not transport truth.\n"
        f"baseline={baseline_design.signature}; candidate={candidate_design.signature}",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_verified_design(
    summary: dict[str, Any],
    mc_summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    baseline_design: ShieldDesign,
    candidate_design: ShieldDesign,
    output_path: Path,
    dpi: int,
) -> None:
    baseline_flux = np.asarray(arrays["baseline_variance_reduced_flux"], dtype=np.float64)
    candidate_flux = np.asarray(arrays["candidate_variance_reduced_flux"], dtype=np.float64)
    baseline_log, candidate_log, log_floor = _normalized_log_pair(baseline_flux, candidate_flux)
    far_mask = np.asarray(arrays["candidate_far_mask"], dtype=bool)
    deep_mask = np.asarray(arrays["candidate_deep_far_mask"], dtype=bool)
    extent = [0.0, baseline_design.grid_size, 0.0, baseline_design.grid_size]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    baseline_image = _plot_layout(axes[0, 0], baseline_design, "Baseline material layout")
    candidate_image = _plot_layout(
        axes[0, 1],
        candidate_design,
        f"Verified candidate {candidate_design.signature}",
    )
    fig.colorbar(
        baseline_image,
        ax=[axes[0, 0], axes[0, 1]],
        label="absorber occupancy",
        shrink=0.8,
    )
    delta = candidate_design.mask().astype(float) - baseline_design.mask().astype(float)
    delta_image = axes[0, 2].imshow(
        delta,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[0, 2].set_title("Compiled edit: red added, blue removed")
    axes[0, 2].set_xlabel("x [cm]")
    axes[0, 2].set_ylabel("y [cm]")
    fig.colorbar(delta_image, ax=axes[0, 2], label="material edit")

    flux_image = None
    for axis, values, title in (
        (axes[1, 0], baseline_log, "Baseline unbiased weight-window MC"),
        (axes[1, 1], candidate_log, "Candidate unbiased weight-window MC"),
    ):
        flux_image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=log_floor,
            vmax=0.0,
            interpolation="nearest",
        )
        _overlay_regions(axis, far_mask, deep_mask, extent)
        axis.set_title(title)
        axis.set_xlabel("x [cm]")
        axis.set_ylabel("y [cm]")
    if flux_image is not None:
        fig.colorbar(
            flux_image,
            ax=[axes[1, 0], axes[1, 1]],
            label="log10 flux, common absolute normalization",
            shrink=0.8,
        )
    handles = [
        Line2D([0], [0], color="lime", linewidth=1.2, label="far-field metric"),
        Line2D([0], [0], color="cyan", linewidth=1.0, linestyle="--", label="deep diagnostic"),
    ]
    axes[1, 0].legend(handles=handles, loc="lower left", fontsize=8)

    ratio = np.log10(
        np.clip(
            candidate_flux / np.maximum(baseline_flux, 1.0e-300),
            1.0e-2,
            1.0e2,
        )
    )
    ratio_image = axes[1, 2].imshow(
        ratio,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    _overlay_regions(axes[1, 2], far_mask, deep_mask, extent)
    axes[1, 2].set_title("Candidate / baseline unbiased VR-MC")
    axes[1, 2].set_xlabel("x [cm]")
    axes[1, 2].set_ylabel("y [cm]")
    fig.colorbar(
        ratio_image,
        ax=axes[1, 2],
        label="log10 flux ratio (clipped to 0.1–10)",
    )

    comparison = mc_summary["design_comparison"]
    far = comparison["far_field"]
    deep = comparison["deep_far_field"]
    fig.suptitle(
        "Verified shield design — independent unbiased Monte Carlo evidence\n"
        f"far-field reduction={100.0 * float(far['estimated_reduction_fraction']):.1f}% "
        f"(z={float(far['difference_z']):.2f}, acceptance gate); "
        f"deep diagnostic reduction={100.0 * float(deep['estimated_reduction_fraction']):.1f}% "
        f"(z={float(deep['difference_z']):.2f}); "
        f"claim={summary.get('final_claim', 'unknown')}",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_mc_certification(
    mc_summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    output_path: Path,
    dpi: int,
) -> None:
    flux_names = (
        "baseline_analog_flux",
        "baseline_variance_reduced_flux",
        "candidate_analog_flux",
        "candidate_variance_reduced_flux",
    )
    fluxes = [np.asarray(arrays[name], dtype=np.float64) for name in flux_names]
    scale = max(max(float(np.max(values)) for values in fluxes), 1.0e-300)
    normalized = [values / scale for values in fluxes]
    positive = np.concatenate([values[values > 0.0] for values in normalized])
    floor = max(float(np.quantile(positive, 0.005)), 1.0e-12)
    log_fluxes = [np.log10(np.maximum(values, floor)) for values in normalized]
    extent = [0.0, 7.0, 0.0, 7.0]
    far_mask = np.asarray(arrays["candidate_far_mask"], dtype=bool)
    deep_mask = np.asarray(arrays["candidate_deep_far_mask"], dtype=bool)
    fig, axes = plt.subplots(3, 3, figsize=(19, 16), constrained_layout=True)

    flux_image = None
    for axis, values, title in (
        (axes[0, 0], log_fluxes[0], "Baseline analog flux"),
        (axes[0, 1], log_fluxes[1], "Baseline weight-window flux"),
        (axes[1, 0], log_fluxes[2], "Candidate analog flux"),
        (axes[1, 1], log_fluxes[3], "Candidate weight-window flux"),
    ):
        flux_image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=math.log10(floor),
            vmax=0.0,
            interpolation="nearest",
        )
        _overlay_regions(axis, far_mask, deep_mask, extent)
        axis.set_title(title)
        axis.set_xlabel("x [cm]")
        axis.set_ylabel("y [cm]")
    if flux_image is not None:
        fig.colorbar(
            flux_image,
            ax=[axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
            label="log10 flux, common absolute normalization",
            shrink=0.72,
        )

    gain_image = None
    for axis, prefix, title in (
        (axes[0, 2], "baseline", "Baseline cellwise error gain"),
        (axes[1, 2], "candidate", "Candidate cellwise error gain"),
    ):
        analog_error = np.asarray(arrays[f"{prefix}_analog_relative_error"], dtype=np.float64)
        vr_error = np.asarray(arrays[f"{prefix}_variance_reduced_relative_error"], dtype=np.float64)
        gain = np.log10(
            np.clip(
                analog_error / np.maximum(vr_error, 1.0e-6),
                1.0e-2,
                1.0e2,
            )
        )
        gain_image = axis.imshow(
            gain,
            origin="lower",
            extent=extent,
            cmap="PiYG",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        _overlay_regions(axis, far_mask, deep_mask, extent)
        axis.set_title(title)
        axis.set_xlabel("x [cm]")
        axis.set_ylabel("y [cm]")
    if gain_image is not None:
        fig.colorbar(
            gain_image,
            ax=[axes[0, 2], axes[1, 2]],
            label="log10(relative error analog / VR)",
            shrink=0.8,
        )

    ratio = np.log10(
        np.clip(
            fluxes[3] / np.maximum(fluxes[1], 1.0e-300),
            1.0e-2,
            1.0e2,
        )
    )
    ratio_image = axes[2, 0].imshow(
        ratio,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    _overlay_regions(axes[2, 0], far_mask, deep_mask, extent)
    axes[2, 0].set_title("Candidate / baseline unbiased VR flux")
    axes[2, 0].set_xlabel("x [cm]")
    axes[2, 0].set_ylabel("y [cm]")
    fig.colorbar(ratio_image, ax=axes[2, 0], label="log10 flux ratio")

    _mc_region_panel(axes[2, 1], mc_summary)

    labels = []
    z_values = []
    colors = []
    for design_name, short_name in (("baseline", "B"), ("candidate", "C")):
        consistency = mc_summary["designs"][design_name]["unbiased_consistency"]
        for region, short_region in (("far_field", "far"), ("deep_far_field", "deep")):
            value = float(consistency[region]["unpaired_consistency_z"])
            labels.append(f"{short_name} {short_region}")
            z_values.append(value)
            colors.append("#54a24b" if abs(value) <= 2.0 else "#e45756")
    axes[2, 2].bar(labels, z_values, color=colors)
    axes[2, 2].axhline(2.0, color="black", linestyle="--", linewidth=1.0)
    axes[2, 2].axhline(-2.0, color="black", linestyle="--", linewidth=1.0)
    axes[2, 2].set_ylabel("Analog − VR consistency z")
    axes[2, 2].set_title("Unbiased-method consistency\nfar field is the acceptance gate")
    axes[2, 2].grid(axis="y", alpha=0.2)

    comparison = mc_summary["design_comparison"]
    far_reduction = float(comparison["far_field"]["estimated_reduction_fraction"])
    deep_reduction = float(comparison["deep_far_field"]["estimated_reduction_fraction"])
    fig.suptitle(
        "Unbiased Monte Carlo certification\n"
        f"far-field reduction={100.0 * far_reduction:.1f}% "
        f"(z={float(comparison['far_field']['difference_z']):.2f}); "
        f"deep diagnostic reduction={100.0 * deep_reduction:.1f}% "
        f"(z={float(comparison['deep_far_field']['difference_z']):.2f})",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_population_control(
    mc_summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    output_path: Path,
    dpi: int,
) -> None:
    extent = [0.0, 7.0, 0.0, 7.0]
    far_mask = np.asarray(arrays["candidate_far_mask"], dtype=bool)
    deep_mask = np.asarray(arrays["candidate_deep_far_mask"], dtype=bool)
    guide_fields = [
        np.asarray(arrays["baseline_guide_flux"], dtype=np.float64),
        np.asarray(arrays["candidate_guide_flux"], dtype=np.float64),
    ]
    guide_scale = max(float(np.max(guide_fields[0])), float(np.max(guide_fields[1])), 1.0e-300)
    guide_positive = np.concatenate(
        [values[values > 0.0] / guide_scale for values in guide_fields]
    )
    guide_floor = max(float(np.quantile(guide_positive, 0.005)), 1.0e-12)
    levels = [
        np.asarray(arrays["baseline_importance_levels"], dtype=np.float64),
        np.asarray(arrays["candidate_importance_levels"], dtype=np.float64),
    ]
    visit_gain = [
        np.log2(
            (np.asarray(arrays[f"{prefix}_variance_reduced_cell_visits"], dtype=np.float64) + 1.0)
            / (np.asarray(arrays[f"{prefix}_analog_cell_visits"], dtype=np.float64) + 1.0)
        )
        for prefix in ("baseline", "candidate")
    ]
    visit_limit = max(
        float(np.quantile(np.abs(np.concatenate([values.ravel() for values in visit_gain])), 0.98)),
        1.0,
    )
    histories = {
        design_name: max(
            int(mc_summary["designs"][design_name]["variance_reduced"]["histories"]),
            1,
        )
        for design_name in ("baseline", "candidate")
    }
    split_fields = [
        np.log10(
            1.0
            + np.asarray(arrays[f"{prefix}_variance_reduced_split_children_map"], dtype=np.float64)
            / histories[prefix]
        )
        for prefix in ("baseline", "candidate")
    ]
    split_ceiling = max(float(np.max(split_fields)), 1.0e-6)
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), constrained_layout=True)
    for row, design_name in enumerate(("baseline", "candidate")):
        guide_image = axes[row, 0].imshow(
            np.log10(np.maximum(guide_fields[row] / guide_scale, guide_floor)),
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=math.log10(guide_floor),
            vmax=0.0,
            interpolation="nearest",
        )
        level_image = axes[row, 1].imshow(
            levels[row],
            origin="lower",
            extent=extent,
            cmap="plasma",
            vmin=0.0,
            vmax=max(float(np.max(levels[0])), float(np.max(levels[1])), 1.0),
            interpolation="nearest",
        )
        visit_image = axes[row, 2].imshow(
            visit_gain[row],
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-visit_limit,
            vmax=visit_limit,
            interpolation="nearest",
        )
        split_image = axes[row, 3].imshow(
            split_fields[row],
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=0.0,
            vmax=split_ceiling,
            interpolation="nearest",
        )
        axes[row, 0].set_title(f"{design_name.capitalize()} guide flux")
        axes[row, 1].set_title(f"{design_name.capitalize()} importance levels")
        axes[row, 2].set_title(f"{design_name.capitalize()} VR / analog visits")
        axes[row, 3].set_title(f"{design_name.capitalize()} split children / root")
        for axis in axes[row]:
            _overlay_regions(axis, far_mask, deep_mask, extent)
            axis.set_xlabel("x [cm]")
            axis.set_ylabel("y [cm]")
    fig.colorbar(guide_image, ax=[axes[0, 0], axes[1, 0]], label="log10 guide flux, common norm", shrink=0.8)
    fig.colorbar(level_image, ax=[axes[0, 1], axes[1, 1]], label="importance level", shrink=0.8)
    fig.colorbar(visit_image, ax=[axes[0, 2], axes[1, 2]], label="log2[(VR visits+1)/(analog visits+1)]", shrink=0.8)
    fig.colorbar(split_image, ax=[axes[0, 3], axes[1, 3]], label="log10(1 + split children/root)", shrink=0.8)

    baseline_vr = mc_summary["designs"]["baseline"]["variance_reduced"]
    candidate_vr = mc_summary["designs"]["candidate"]["variance_reduced"]
    fig.suptitle(
        "Weight-window population-control audit\n"
        f"transported particles/root: baseline={float(baseline_vr['transported_particles_per_root']):.2f}, "
        f"candidate={float(candidate_vr['transported_particles_per_root']):.2f}; "
        f"split-cap hits: baseline={int(baseline_vr['split_cap_hits'])}, "
        f"candidate={int(candidate_vr['split_cap_hits'])}",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_autonomous_shield_figures(
    summary: dict[str, Any],
    output_dir: str | Path,
    trajectory_path: str | Path | None = None,
    environment: ShieldToolEnvironment | None = None,
    dpi: int = 240,
    response_fields: dict[str, Any] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    trajectory = Path(trajectory_path) if trajectory_path is not None else None
    events = _read_trajectory(trajectory)
    written: dict[str, str] = {}
    selected = _selected_payload(summary)
    selected_signature = str(selected.get("signature")) if selected is not None else None
    resolved_response_fields = response_fields
    if resolved_response_fields is None:
        resolved_response_fields = _runtime_response_fields(environment, selected_signature)
    if resolved_response_fields is None:
        resolved_response_fields = _load_response_fields(output_dir)
    if resolved_response_fields is not None:
        metadata = resolved_response_fields.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            resolved_response_fields["metadata"] = metadata
        if not metadata.get("discretization"):
            metadata["discretization"] = _response_discretization_from_config(output_dir)
        _write_response_fields(resolved_response_fields, output_dir)

    dashboard = output_dir / "agent_search_dashboard.png"
    _plot_search_dashboard(summary, events, dashboard, dpi)
    written["search_dashboard"] = str(dashboard)

    timeline = output_dir / "agent_decision_timeline.png"
    if _plot_decision_timeline(summary, events, timeline, dpi):
        written["decision_timeline"] = str(timeline)

    evidence_path = output_dir / "candidate_evidence_chain.png"
    if _plot_candidate_evidence(
        summary,
        events,
        output_dir,
        evidence_path,
        environment,
        resolved_response_fields,
        dpi,
    ):
        written["candidate_evidence"] = str(evidence_path)

    baseline_design = environment.baseline_design if environment is not None else baseline_lattice_design()
    candidate_design = (
        environment.candidate(selected_signature).design
        if environment is not None
        and selected_signature is not None
        and environment.candidate(selected_signature) is not None
        else _design_from_payload(selected, baseline_design)
    )
    storyboard = output_dir / "gmc_discovery_storyboard.png"
    if _plot_gmc_discovery_storyboard(
        summary,
        events,
        resolved_response_fields,
        baseline_design,
        candidate_design,
        storyboard,
        dpi,
    ):
        written["gmc_discovery_storyboard"] = str(storyboard)
    animation = output_dir / "gmc_screening_reel.gif"
    if _plot_gmc_screening_animation(
        summary,
        events,
        baseline_design,
        animation,
        dpi,
    ):
        written["gmc_screening_animation"] = str(animation)
    response_comparison = output_dir / "response_method_comparison.png"
    if _plot_response_comparison(
        resolved_response_fields,
        baseline_design,
        candidate_design,
        response_comparison,
        dpi,
    ):
        written["response_comparison"] = str(response_comparison)

    mc_summary, arrays = _mc_artifacts(selected, output_dir)
    if mc_summary is not None and arrays is not None:
        if summary.get("final_claim") == "verified_improvement":
            verified_path = output_dir / "verified_design.png"
            _plot_verified_design(
                summary,
                mc_summary,
                arrays,
                baseline_design,
                candidate_design,
                verified_path,
                dpi,
            )
            written["verified_design"] = str(verified_path)
        certification = output_dir / "mc_certification.png"
        _plot_mc_certification(mc_summary, arrays, certification, dpi)
        written["mc_certification"] = str(certification)
        population = output_dir / "mc_population_control.png"
        _plot_population_control(mc_summary, arrays, population, dpi)
        written["mc_population_control"] = str(population)
    return written
