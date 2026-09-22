# GenVR-Fusion run summary

**Scientific claim (scoped):** This predeclared lattice run records a baseline and refinement under a fixed dense reference; the isolated position-versus-angle mechanism was not resolved by the supplied action set.

Common reference: `angular_refined:oracle`.

| Experiment | States | CFM ray vs dense | Exact-MC ray vs dense | CFM L1 vs common | Process seconds |
|---|---:|---:|---:|---:|---:|
| coarse | 32 | 0.00442353 | 0.00525156 | 0.0509805 | 7.451 |
| angular_refined | 128 | 0.00257258 | 0.00265411 | 0.0344021 | 6.368 |

## Interpretation

- Stronger isolated allocation at equal state budget: **not resolved**.
- Combined exact-MC ray-index reduction vs fixed dense: **49.5%**.
- Combined CFM ray-index reduction vs fixed dense: **41.8%**.
- Combined CFM normalized L1 vs its exact-MC response: **3.4%**.
- All causal trials were retained; no best-of-N selection was used.
- See `trajectory.jsonl` for every observation, action, tool call, feedback event, and decision.

## Limitations

- The spectral ray index is a diagnostic proxy, not a universal physical observable.
- The local exact-MC response operators use finite samples and remain seed sensitive.
- The dense S_N field is a fixed deterministic comparison, not an exact continuum solution.
- The study does not establish novelty relative to every published ray-effect remedy.
- The causal ablation uses one declared seed; the separate random-reference trials expose seed sensitivity but are not matched repeated trials.
