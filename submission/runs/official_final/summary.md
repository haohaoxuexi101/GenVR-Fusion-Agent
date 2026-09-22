# RayLab-GMC run summary

**Scientific claim (scoped):** In this single-seed, predeclared lattice ablation, angular allocation reduced the CFM-versus-dense spectral ray index by 43.5% relative to position allocation at the same 128-state budget. Joint refinement reduced the local exact-MC-versus-dense ray index by 57.3% relative to the coarse case, while the refined CFM retained 5.0% normalized L1 error versus its local exact-MC response; discretization refinement therefore mitigates but does not by itself eliminate the learned-operator residual.

Common reference: `combined_refined:oracle`.

| Experiment | States | CFM ray vs dense | Exact-MC ray vs dense | CFM L1 vs common | Process seconds |
|---|---:|---:|---:|---:|---:|
| coarse | 32 | 0.00384273 | 0.00213594 | 0.0767723 | 7.672 |
| position_refined | 128 | 0.00443073 | 0.00457611 | 0.0769775 | 10.721 |
| angular_refined | 128 | 0.00250119 | 0.00216873 | 0.0619746 | 10.837 |
| combined_refined | 512 | 0.00312556 | 0.000911311 | 0.0503924 | 48.936 |

## Interpretation

- Stronger isolated allocation at equal state budget: **angular discretization**.
- Combined exact-MC ray-index reduction vs fixed dense: **57.3%**.
- Combined CFM ray-index reduction vs fixed dense: **18.7%**.
- Combined CFM normalized L1 vs its exact-MC response: **5.0%**.
- All causal trials were retained; no best-of-N selection was used.
- See `trajectory.jsonl` for every observation, action, tool call, feedback event, and decision.

## Limitations

- The spectral ray index is a diagnostic proxy, not a universal physical observable.
- The local exact-MC response operators use finite samples and remain seed sensitive.
- The dense S_N field is a fixed deterministic comparison, not an exact continuum solution.
- The study does not establish novelty relative to every published ray-effect remedy.
- The causal ablation uses one declared seed; the separate random-reference trials expose seed sensitivity but are not matched repeated trials.
