from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "submission" / "runs" / "official_llm"
SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    errors: list[str] = []
    required = [
        ROOT / "README.md",
        ROOT / "environment.yml",
        ROOT / "LICENSE",
        ROOT / "submission" / "SCIENTIFIC_FINDING_AND_ENVIRONMENT.md",
        ROOT / "submission" / "REFERENCE_BENCHMARK_DESIGN.md",
        ROOT / "submission" / "EVIDENCE_LEDGER.md",
        ROOT / "submission" / "TRACEABILITY.md",
        ROOT / "submission" / "API_AND_SECURITY.md",
        ROOT / "submission" / "SUBMISSION_MANIFEST.md",
        ROOT / "submission" / "api_validation" / "deepseek_model_matrix.json",
        ROOT / "scripts" / "demo_one_minute.ps1",
        RUN / "config.json",
        RUN / "run_manifest.json",
        RUN / "postprocess_manifest.json",
        RUN / "trajectory.jsonl",
        RUN / "summary.json",
        RUN / "summary.md",
        RUN / "ablation_summary.png",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty: {path.relative_to(ROOT)}")

    records: list[dict] = []
    trajectory = RUN / "trajectory.jsonl"
    if trajectory.is_file():
        for line_no, line in enumerate(trajectory.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid trajectory JSON line {line_no}: {exc}")
                continue
            missing = {"schema_version", "run_id", "timestamp_utc", "event_type"} - set(record)
            if missing:
                errors.append(f"trajectory line {line_no} missing keys: {sorted(missing)}")
            records.append(record)
    event_counts = Counter(record.get("event_type") for record in records)
    expected_events = {
        "run_started",
        "decision",
        "agent_input",
        "agent_output",
        "tool_call",
        "tool_result",
        "environment_feedback",
        "llm_exchange",
        "run_completed",
        "summary_regenerated",
    }
    for event_type in expected_events:
        if event_counts[event_type] == 0:
            errors.append(f"trajectory missing event type: {event_type}")

    if (RUN / "config.json").is_file():
        config = load_json(RUN / "config.json")
        expected_ids = [x["experiment_id"] for x in config.get("actions", [])]
        random_dirs = sorted((RUN / "experiments").glob("random_*"))
        for experiment_id in expected_ids + [path.name for path in random_dirs]:
            exp_dir = RUN / "experiments" / experiment_id
            for name in [
                "diagnostics.json",
                "metrics.json",
                f"{config['case']}_stage8_ultra.npz",
                f"{config['case']}_stage8_ultra.png",
                "stdout.txt",
                "stderr.txt",
            ]:
                if not (exp_dir / name).is_file():
                    errors.append(f"missing experiment artifact: {(exp_dir / name).relative_to(ROOT)}")

    for manifest_name, base in [("run_manifest.json", ROOT), ("postprocess_manifest.json", RUN)]:
        path = RUN / manifest_name
        if not path.is_file():
            continue
        manifest = load_json(path)
        groups = [(manifest.get("files", {}), base)]
        if manifest_name == "postprocess_manifest.json":
            groups = [(manifest.get("source_files", {}), ROOT), (manifest.get("artifacts", {}), RUN)]
        for group, group_base in groups:
            for relative, metadata in group.items():
                candidate = group_base / Path(relative)
                if not candidate.is_file():
                    errors.append(f"manifest target missing: {relative}")
                elif sha256(candidate) != metadata.get("sha256"):
                    errors.append(f"manifest hash mismatch: {relative}")

    summary_path = RUN / "summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        regenerated = [x for x in records if x.get("event_type") == "summary_regenerated"]
        if regenerated and regenerated[-1].get("scientific_claim") != summary.get("scientific_claim"):
            errors.append("latest trajectory scientific claim does not match summary.json")
        causal = summary.get("causal_experiments", [])
        if len(causal) != 4:
            errors.append(f"expected 4 causal experiments, found {len(causal)}")
        if len(summary.get("random_reference_experiments", [])) != 3:
            errors.append("expected 3 random reference experiments")

    api_validation_path = ROOT / "submission" / "api_validation" / "deepseek_model_matrix.json"
    if api_validation_path.is_file():
        api_validation = load_json(api_validation_path)
        if api_validation.get("status") != "PASS":
            errors.append("DeepSeek model-matrix validation did not pass")
        validated_models = {item.get("model") for item in api_validation.get("models", []) if item.get("status") == "PASS"}
        if not {"deepseek-v4-flash", "deepseek-v4-pro"}.issubset(validated_models):
            errors.append("DeepSeek V4-Flash and V4-Pro were not both validated")
        if api_validation.get("secret_recorded") is not False:
            errors.append("API validation does not explicitly attest secret_recorded=false")

    run_manifest_path = RUN / "run_manifest.json"
    if run_manifest_path.is_file():
        run_manifest = load_json(run_manifest_path)
        if run_manifest.get("policy") != "deepseek":
            errors.append("official run manifest policy is not deepseek")
    if event_counts["llm_exchange"] < 1:
        errors.append("official trajectory has no recorded LLM exchange")

    scan_roots = [ROOT / "ray_agent", ROOT / "scripts", ROOT / "configs", ROOT / "submission"]
    text_suffixes = {".py", ".md", ".json", ".jsonl", ".yml", ".yaml", ".txt", ".ps1", ".sh", ".cff", ".example"}
    secret_hits = 0
    for scan_root in scan_roots:
        for path in scan_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in text_suffixes:
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                secret_hits += len(SECRET_PATTERN.findall(text))
    if secret_hits:
        errors.append(f"secret scan detected {secret_hits} possible API key value(s); values are intentionally not printed")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "required_files_checked": len(required),
        "trajectory_records": len(records),
        "event_counts": dict(sorted(event_counts.items())),
        "secret_hits": secret_hits,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
