# Analysis plan: hypotheses and decision rule

The hypotheses and the analysis plan below were fixed before the confirmatory ten-seed run.
Outcomes are reported in the manuscript (hypothesis table) and in `results/master_results.json`.

## Hypotheses
- **H1.** The attribution family forms a faithfulness tier above Grad-CAM on the
  leakage-controlled removal metric (Axis A).
- **H2.** Grad-CAM is statistically indistinguishable from a random baseline on Axis A.
- **H3.** Methods occupy distinct positions on a faithfulness-by-clinical-validity profile,
  with the family separating from Grad-CAM on both axes.
- **H4.** Under a controlled shortcut injection, only methods that are faithful on Axis A
  localize and detect the injected artefact.

## Decision rule (as fixed before the confirmatory run)
A method passes if it is faithful on Axis A where the removal metric is powered and is
randomization-sensitive on Axis C (across-seed aggregate mean_seeds(|C|) < τ_C = 0.30, fixed
a priori). The shortcut-injection test provides ground-truth confirmation.

## Fixed settings
- τ_C = 0.30 (Axis C).
- Seeds: [42, 1, 7, 2, 3, 4, 5, 6, 8, 9].
- Removal fractions 0–0.50 in steps of 0.05; neighbourhood imputer k = 8, guard band g = 15.

## Refinements made during peer review
1. **Axis C** is reported as a diagnostic and is no longer part of the decision rule, because a
   random map run through the same pipeline also passes it (mean_seeds(|C|) ≈ 0.001).
2. **H2** is tested by equivalence (TOST) at a margin of ±0.02 ABPC, with ±0.01 and ±0.03 as
   sensitivity margins; classes on which no method separates from random (MI, CD) are not
   counted as evidence.
3. **Shortcut detection** uses a threshold calibrated on 120 records and evaluated on 200
   disjoint records; a method is judged to expose the shortcut when the lower 95% Wilson bound
   of its detection rate is at least 80% in every tested setting.
4. **Terminology.** "Family" denotes the four input-multiplied methods (Grad×Input, DeepLIFT,
   Integrated Gradients, GradientSHAP); Saliency is analysed separately. The SHAP-type method is
   implemented with Captum `GradientShap` and is named GradientSHAP.
