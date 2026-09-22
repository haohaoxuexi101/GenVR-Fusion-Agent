from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "gmc_material_discovery" / "run_20260920_020815"
REPOSITORY_URL = "https://github.com/haohaoxuexi101/GenVR-Fusion-Agent"
RELEASE_TAG = "v0.1.0"
RELEASE_COMMIT = "ca885735e09cb46a6afba29c1a47f19d0fd8eda1"
SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    errors: list[str] = []
    required = [
        ROOT / "README.md",
        ROOT / "environment.yml",
        ROOT / "environment_cpu.yml",
        ROOT / "requirements.txt",
        ROOT / "LICENSE",
        ROOT / "CITATION.cff",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "configs" / "gmc_material_discovery_agent.json",
        ROOT / "scripts" / "run_gmc_material_discovery_agent.sh",
        ROOT / "scripts" / "25_run_autonomous_shield_agent.py",
        ROOT / "scripts" / "26_plot_autonomous_shield_agent.py",
        ROOT / "scripts" / "27_build_scientific_finding_report.py",
        ROOT / "submission" / "SCIENTIFIC_FINDING_AND_ENVIRONMENT.md",
        ROOT / "submission" / "REFERENCE_BENCHMARK_DESIGN.md",
        ROOT / "submission" / "EVIDENCE_LEDGER.md",
        ROOT / "submission" / "TRACEABILITY.md",
        ROOT / "submission" / "API_AND_SECURITY.md",
        ROOT / "submission" / "SUBMISSION_MANIFEST.md",
        ROOT / "submission" / "assets" / "api_participation_audit.png",
        ROOT / "output" / "pdf" / "GenShield-Agent_科学发现与环境定义报告.pdf",
        RUN / "config.json",
        RUN / "run_manifest.json",
        RUN / "trajectory.jsonl",
        RUN / "summary.json",
        RUN / "summary.md",
        RUN / "gmc_screening" / "manifest.json",
        RUN / "gmc_screening_reel.gif",
        RUN / "agent_search_dashboard.png",
        RUN / "agent_decision_timeline.png",
        RUN / "candidate_evidence_chain.png",
        RUN / "gmc_discovery_storyboard.png",
        RUN / "mc_certification.png",
        RUN / "mc_population_control.png",
        RUN / "mc" / "01_d7d960548ee7" / "certification.json",
        RUN / "mc" / "01_d7d960548ee7" / "fields.npz",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty: {path.relative_to(ROOT)}")

    trajectory_records: list[dict] = []
    trajectory_path = RUN / "trajectory.jsonl"
    if trajectory_path.is_file():
        for line_number, line in enumerate(
            trajectory_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid trajectory JSON line {line_number}: {exc}")
                continue
            missing = {"schema_version", "run_id", "timestamp_utc", "event_type"} - set(record)
            if missing:
                errors.append(f"trajectory line {line_number} missing keys: {sorted(missing)}")
            trajectory_records.append(record)

    event_counts = Counter(record.get("event_type") for record in trajectory_records)
    expected_counts = {
        "run_started": 1,
        "llm_exchange": 17,
        "agent_input": 12,
        "agent_output": 12,
        "decision": 12,
        "tool_call": 12,
        "tool_result": 12,
        "environment_feedback": 11,
        "verifier_decision": 2,
        "run_completed": 1,
        "report_generated": 1,
    }
    for event_type, expected in expected_counts.items():
        observed = event_counts[event_type]
        if observed != expected:
            errors.append(f"trajectory {event_type}: expected {expected}, observed {observed}")

    exchanges = [
        record for record in trajectory_records if record.get("event_type") == "llm_exchange"
    ]
    response_ids = {record.get("response_id") for record in exchanges}
    total_tokens = sum(
        int((record.get("usage") or {}).get("total_tokens", 0)) for record in exchanges
    )
    if len(response_ids) != 17:
        errors.append(f"expected 17 unique response_id values, observed {len(response_ids)}")
    if total_tokens != 233_244:
        errors.append(f"expected 233244 API tokens, observed {total_tokens}")
    if sum(bool(record.get("accepted")) for record in exchanges) != 12:
        errors.append("expected 12 accepted API responses")

    manifest_path = RUN / "run_manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        if manifest.get("model") != "deepseek-v4-flash":
            errors.append("formal model is not deepseek-v4-flash")
        if manifest.get("llm_required") is not True:
            errors.append("formal manifest does not require the LLM")
        if manifest.get("deterministic_policy_fallback") is not False:
            errors.append("formal manifest permits deterministic fallback")
        if manifest.get("human_intervention") != "None after launch.":
            errors.append("formal manifest does not attest zero post-launch intervention")
        if manifest.get("discovery_mode") != "gmc_only":
            errors.append("formal discovery mode is not gmc_only")
        if manifest.get("diffusion_model_used") is not False:
            errors.append("formal run unexpectedly used a diffusion model")
        for relative, metadata in manifest.get("files", {}).items():
            candidate = ROOT / relative
            if not candidate.is_file():
                errors.append(f"manifest target missing: {relative}")
            elif sha256(candidate) != metadata.get("sha256"):
                errors.append(f"manifest hash mismatch: {relative}")

    summary_path = RUN / "summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        expected_summary = {
            "final_claim": "inconclusive",
            "steps_completed": 12,
            "response_failures": 5,
            "discovery_mode": "gmc_only",
            "deterministic_policy_fallback": False,
        }
        for key, expected in expected_summary.items():
            if summary.get(key) != expected:
                errors.append(f"summary {key}: expected {expected!r}, observed {summary.get(key)!r}")
        progress = summary.get("discovery_progress", {})
        progress_expected = {
            "generated_structures": 59,
            "gmc_screened_structures": 16,
            "gmc_guided_and_screened": 12,
        }
        for key, expected in progress_expected.items():
            if progress.get(key) != expected:
                errors.append(f"discovery_progress {key}: expected {expected}, observed {progress.get(key)}")
        budget = summary.get("budget_usage", {})
        if (budget.get("mc_root_histories") or {}).get("used") != 80_000:
            errors.append("formal run did not use the recorded 80000 MC root histories")

    certification_path = RUN / "mc" / "01_d7d960548ee7" / "certification.json"
    if certification_path.is_file():
        certification = load_json(certification_path)
        far = certification.get("design_comparison", {}).get("far_field", {})
        ratio = far.get("candidate_over_baseline")
        difference_z = far.get("difference_z")
        baseline_z = (
            certification.get("designs", {})
            .get("baseline", {})
            .get("unbiased_consistency", {})
            .get("far_field", {})
            .get("unpaired_consistency_z")
        )
        if ratio is None or abs(float(ratio) - 0.10998630215249551) > 1.0e-12:
            errors.append(f"unexpected far-field candidate/baseline ratio: {ratio}")
        if difference_z is None or abs(float(difference_z) + 50.802208627497215) > 1.0e-9:
            errors.append(f"unexpected far-field difference z: {difference_z}")
        if baseline_z is None or abs(float(baseline_z) + 2.1476675804908627) > 1.0e-9:
            errors.append(f"unexpected baseline consistency z: {baseline_z}")

    report_source = ROOT / "submission" / "SCIENTIFIC_FINDING_AND_ENVIRONMENT.md"
    if report_source.is_file():
        report_text = report_source.read_text(encoding="utf-8")
        for required_text in (
            REPOSITORY_URL,
            "小手震",
            "不使用外部训练数据集",
            RELEASE_TAG,
            RELEASE_COMMIT,
            "inconclusive",
        ):
            if required_text not in report_text:
                errors.append(f"report is missing required disclosure: {required_text}")

    primary_pdf = ROOT / "output" / "pdf" / "GenShield-Agent_科学发现与环境定义报告.pdf"
    legacy_pdf = ROOT / "output" / "pdf" / "GenVR-Fusion_科学发现与环境定义报告.pdf"
    if primary_pdf.is_file() and legacy_pdf.is_file() and sha256(primary_pdf) != sha256(legacy_pdf):
        errors.append("legacy PDF alias does not match the current report")

    remote = git_output("remote", "get-url", "origin")
    if remote not in {REPOSITORY_URL, f"{REPOSITORY_URL}.git"}:
        errors.append(f"unexpected origin URL: {remote}")
    tag_commit = git_output("rev-parse", f"{RELEASE_TAG}^{{}}")
    if tag_commit != RELEASE_COMMIT:
        errors.append(f"tag {RELEASE_TAG} does not resolve to {RELEASE_COMMIT}")

    scan_paths = [
        ROOT / "README.md",
        ROOT / "CITATION.cff",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "ray_agent",
        ROOT / "scripts",
        ROOT / "configs",
        ROOT / "submission" / "SCIENTIFIC_FINDING_AND_ENVIRONMENT.md",
        ROOT / "submission" / "REFERENCE_BENCHMARK_DESIGN.md",
        ROOT / "submission" / "EVIDENCE_LEDGER.md",
        ROOT / "submission" / "TRACEABILITY.md",
        ROOT / "submission" / "API_AND_SECURITY.md",
        ROOT / "submission" / "SUBMISSION_MANIFEST.md",
        RUN,
    ]
    text_suffixes = {
        ".py",
        ".md",
        ".json",
        ".jsonl",
        ".yml",
        ".yaml",
        ".txt",
        ".ps1",
        ".sh",
        ".cff",
        ".example",
    }
    secret_hits: list[str] = []
    for scan_path in scan_paths:
        candidates = [scan_path] if scan_path.is_file() else scan_path.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in text_suffixes:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if SECRET_PATTERN.search(text):
                secret_hits.append(str(path.relative_to(ROOT)))
    if secret_hits:
        errors.append(f"secret scan detected possible API keys in: {sorted(set(secret_hits))}")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "repository": REPOSITORY_URL,
        "release_tag": RELEASE_TAG,
        "release_commit": RELEASE_COMMIT,
        "required_files_checked": len(required),
        "trajectory_records": len(trajectory_records),
        "event_counts": dict(sorted(event_counts.items())),
        "api_total_tokens": total_tokens,
        "secret_hits": len(secret_hits),
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
