# GenVR-Fusion run summary

**Scientific claim (scoped):** This predeclared lattice run records a baseline and refinement under a fixed dense reference; the isolated position-versus-angle mechanism was not resolved by the supplied action set.

Common reference: `combined_refined:oracle`.

| Experiment | States | CFM ray vs dense | Exact-MC ray vs dense | CFM L1 vs common | Process seconds |
|---|---:|---:|---:|---:|---:|
| coarse | 32 | 0.00442353 | 0.00525156 | 0.0490096 | 6.262 |
| combined_refined | 512 | 0.00462209 | 0.00483455 | 0.0407195 | 8.651 |

## Interpretation

- Stronger isolated allocation at equal state budget: **not resolved**.
- Combined exact-MC ray-index reduction vs fixed dense: **7.9%**.
- Combined CFM ray-index reduction vs fixed dense: **-4.5%**.
- Combined CFM normalized L1 vs its exact-MC response: **4.1%**.
- All causal trials were retained; no best-of-N selection was used.
- See `trajectory.jsonl` for every observation, action, tool call, feedback event, and decision.

## Limitations

- The spectral ray index is a diagnostic proxy, not a universal physical observable.
- The local exact-MC response operators use finite samples and remain seed sensitive.
- The dense S_N field is a fixed deterministic comparison, not an exact continuum solution.
- The study does not establish novelty relative to every published ray-effect remedy.
- The causal ablation uses one declared seed; the separate random-reference trials expose seed sensitivity but are not matched repeated trials.
