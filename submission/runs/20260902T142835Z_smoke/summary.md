# RayLab-GMC run summary

**Scientific claim (scoped):** In this predeclared lattice ablation, not resolved was the stronger isolated contributor under the tested settings; the combined refinement changed the common-reference ray index by 45.8% relative to the coarse baseline.

Common reference: `position_refined:oracle`.

| Experiment | States | Ray index | L1 vs common | Process seconds |
|---|---:|---:|---:|---:|
| coarse | 32 | 0.00318694 | 0.0548516 | 5.990 |
| position_refined | 128 | 0.00172718 | 0.0302084 | 6.221 |

## Interpretation

- Dominant isolated mechanism in this run: **not resolved**.
- Combined-refinement ray-index change relative to coarse baseline: **45.8%**.
- All causal trials were retained; no best-of-N selection was used.
- See `trajectory.jsonl` for every observation, action, tool call, feedback event, and decision.

## Limitations

- The spectral ray index is a diagnostic proxy, not a universal physical observable.
- The final high-resolution local exact-MC response is a finite-sample reference.
- The study does not establish novelty relative to every published ray-effect remedy.
- No result was selected from repeated runs; broader multi-seed confirmation remains future work unless listed in this run.
