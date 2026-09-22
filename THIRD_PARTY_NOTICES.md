# Third-party notices and provenance checklist

The repository uses NumPy, SciPy, Matplotlib, Numba, PyTorch and pytest under their respective licenses. Their versions are recorded in `environment.yml`; the current formal Agent runtime is recorded in `outputs/gmc_material_discovery/run_20260920_020815/run_manifest.json` and is distributed with the report-and-log submission package.

All scientific data used by the project are team-generated. Boundary-response samples are produced by `scripts/01_generate_boundary_data.py` and internal-response samples by `scripts/08_generate_internal_data.py`; the underlying benchmark and random-flight generators are implemented in `gmc/benchmarks2d.py` and `gmc/mc_cell.py`. The CFM models are trained with `scripts/02_train_boundary_cfm.py`, `scripts/09_train_internal_cfm.py`, and the resumable paper-scale pipeline in `scripts/32_train_cfm_resumable.py`. No external or third-party training dataset is used.

The production CFM checkpoints, GMC response caches, candidate flux fields, and Monte Carlo certification arrays are derived from those self-generated data. Large checkpoints and response caches are reproducible intermediate artifacts rather than externally sourced data; formal-run hashes and exact paths are frozen in `outputs/gmc_material_discovery/run_20260920_020815/run_manifest.json` and the accompanying submission package.

The benchmark implementation and agent code added for this work are distributed under the repository MIT License at https://github.com/haohaoxuexi101/GenVR-Fusion-Agent. Scientific papers are cited as literature only; no paper text or figure is redistributed.
