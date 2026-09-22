# GenVR numerical-artifact attribution report

**Scientific claim (scoped):** For the declared lattice experiment family: CFM and projected local-MC respond similarly to interface-angle rotation, indicating a shared finite interface-projection artifact rather than a learned-kernel artifact. No configuration is certified because the required robustness evidence or acceptance gates are incomplete.

The CFM–S_N directional residual is reported only as cross-method disagreement; it is not optimized as an intrinsic CFM ray score.

## Candidate evidence

| Candidate | States | Learning L1 | Rotation L1 | Angular gap | Repeat L1 | MC integral error | Eligible | Seconds |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|
| p1_m2_f16_direct | 32 | 0.084889 | 0.13485 | 0.051548 | 0.041896 | 0.02179 | no | 12.311 |
| p4_m2_f16_direct | 128 | 0.082243 | 0.076607 | 0.012477 | 0.043355 | 0.064561 | no | 32.459 |
| p1_m2_f64_direct | 128 | 0.068535 | 0.031011 | 0.051548 | 0.045347 | 0.079989 | no | 31.004 |
| p4_m2_f64_direct | 512 | 0.066007 | 0.058802 | 0.012477 | 0.043157 | 0.05474 | no | 162.080 |

## Controlled comparisons

| Type | Baseline | Intervention | CFM L1 | Projected-MC L1 | Dense S_N L1 |
|---|---|---|---:|---:|---:|
| interface_rotation | coarse_baseline | coarse_interface_rotation | 0.13485 | 0.1577 | 0 |
| dense_rotation | coarse_baseline | coarse_dense_rotation | 0 | 0 | 0.0067872 |
| repeatability | coarse_baseline | coarse_repeat | 0.041896 | 0.035517 | 0 |
| interface_position_refinement | coarse_baseline | position_refined | 0.04 | 0.057556 | 0 |
| interface_angular_refinement | coarse_baseline | angular_refined | 0.056426 | 0.083584 | 0 |
| dense_angular_refinement | coarse_baseline | dense_angular_refined | 0 | 0 | 0.0050204 |
| global_mc_sampling_refinement | coarse_baseline | coarse_certification | 0 | 0 | 0 |
| interface_position_refinement | coarse_interface_rotation | position_refined_rotation | 0.081486 | 0.05822 | 0 |
| interface_angular_refinement | coarse_interface_rotation | angular_refined_rotation | 0.072952 | 0.046251 | 0 |
| dense_angular_refinement | coarse_dense_rotation | dense_angular_refined_rotation | 0 | 0 | 0.0047881 |
| interface_position_refinement | coarse_repeat | position_refined_repeat | 0.039358 | 0.068085 | 0 |
| interface_angular_refinement | coarse_repeat | angular_refined_repeat | 0.051548 | 0.088641 | 0 |
| interface_rotation | position_refined | position_refined_rotation | 0.076607 | 0.14187 | 0 |
| repeatability | position_refined | position_refined_repeat | 0.043355 | 0.020787 | 0 |
| interface_angular_refinement | position_refined | combined_refined | 0.013205 | 0.034656 | 0 |
| global_mc_sampling_refinement | position_refined | position_refined_certification | 0 | 0 | 0 |
| interface_angular_refinement | position_refined_rotation | combined_refined_rotation | 0.024798 | 0.048315 | 0 |
| interface_angular_refinement | position_refined_repeat | combined_refined_repeat | 0.012477 | 0.033679 | 0 |
| interface_rotation | angular_refined | angular_refined_rotation | 0.031011 | 0.048741 | 0 |
| repeatability | angular_refined | angular_refined_repeat | 0.045347 | 0.034975 | 0 |
| interface_position_refinement | angular_refined | combined_refined | 0.073725 | 0.086816 | 0 |
| global_mc_sampling_refinement | angular_refined | angular_refined_certification | 0 | 0 | 0 |
| interface_position_refinement | angular_refined_rotation | combined_refined_rotation | 0.049884 | 0.057818 | 0 |
| interface_position_refinement | angular_refined_repeat | combined_refined_repeat | 0.071926 | 0.099377 | 0 |
| interface_rotation | combined_refined | combined_refined_rotation | 0.058802 | 0.088578 | 0 |
| repeatability | combined_refined | combined_refined_repeat | 0.043157 | 0.024776 | 0 |
| global_mc_sampling_refinement | combined_refined | combined_certification | 0 | 0 | 0 |
| dense_rotation | dense_angular_refined | dense_angular_refined_rotation | 0 | 0 | 0.0018243 |
| interface_position_refinement | coarse_certification | position_refined_certification | 0.04 | 0.057556 | 0 |
| interface_angular_refinement | coarse_certification | angular_refined_certification | 0.056426 | 0.083584 | 0 |
| interface_angular_refinement | position_refined_certification | combined_certification | 0.013205 | 0.034656 | 0 |
| interface_position_refinement | angular_refined_certification | combined_certification | 0.073725 | 0.086816 | 0 |

## Limitations

- A directional residual between two fields does not identify which field contains the artifact.
- Rotation sensitivity is a causal numerical diagnostic, not a universal physical observable.
- Projected local-MC shares the same finite interface phase space and isolates learned-kernel error only.
- Dense S_N is a finite-angle deterministic diagnostic and is not treated as continuum truth.
- Global history MC is a physical certification reference only at statistically supported tallies.
- A recommendation is withheld unless matched rotation evidence and all declared acceptance gates pass.

See `trajectory.jsonl` for every action, tool call, observation, and policy decision.
