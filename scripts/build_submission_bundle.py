from __future__ import annotations

from pathlib import Path
import hashlib
import json
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
RUN = SUBMISSION / "runs" / "official_final"
ARCHIVE = SUBMISSION / "OPENxxx_RayLab-GMC_报告与日志_非PDF版.zip"
MANIFEST = SUBMISSION / "bundle_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    roots = [
        ROOT / "README.md",
        ROOT / "environment.yml",
        ROOT / "environment_cpu.yml",
        ROOT / "requirements.txt",
        ROOT / ".env.example",
        ROOT / "LICENSE",
        ROOT / "CITATION.cff",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "configs" / "ray_agent_smoke.json",
        ROOT / "configs" / "ray_agent_core.json",
    ]
    files = [path for path in roots if path.is_file()]
    files.extend(sorted(path for path in SUBMISSION.glob("*.md") if path.is_file()))
    files.extend(sorted(path for path in RUN.rglob("*") if path.is_file()))
    files = sorted(set(files), key=lambda path: str(path.relative_to(ROOT)))
    manifest = {
        "schema_version": "1.0",
        "archive_name": ARCHIVE.name,
        "note": "Non-PDF draft bundle. Replace OPENxxx, then add the required PPT and PDF before final submission.",
        "files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(MANIFEST)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, arcname=str(path.relative_to(ROOT)).replace("\\", "/"))
    print(json.dumps({
        "archive": str(ARCHIVE),
        "bytes": ARCHIVE.stat().st_size,
        "sha256": sha256(ARCHIVE),
        "files": len(files),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
