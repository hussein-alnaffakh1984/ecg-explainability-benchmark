# Pre-registration: hypotheses and decision rule

All analyses below were specified before the confirmatory 10-seed run.

## Pre-registered hypotheses
- **H1.** The attribution family (Saliency, Grad×Input, DeepLIFT, Integrated Gradients,
  DeepSHAP) forms a faithfulness tier above Grad-CAM (block separation).
  *Outcome: supported on NORM, HYP and STTC; inconclusive on MI; not supported on CD,
  where the family is at or below random.*
- **H2.** Grad-CAM is statistically indistinguishable from a random baseline under the
  leakage-controlled removal metric. *Outcome: supported in every localizable class
  (Grad-CAM within 0.06 ABPC of random).*
- **H3.** Methods separate on a faithfulness-by-clinical-validity profile (Pareto), with
  no method dominating both axes. *Outcome: supported; Axis-C seed-instability reported.*
- **H4.** Under a calibrated shortcut injection, only faithful methods localize and detect
  the injected artefact. *Outcome: supported; the four family methods detect at 99–100%,
  Saliency and Grad-CAM at 0%.*

## Decision rule
A method is provisionally trustworthy if it passes Axis A (faithful where the removal
metric is powered) AND Axis C (randomization-sensitive on the across-seed aggregate
mean_seeds(|C|) < τ_C = 0.30, fixed a priori). The calibrated shortcut-injection test
(separation AUROC = 1.00, τ_audit = 0.161) provides the decisive ground truth.

## Fixed thresholds
- τ_C = 0.30 (randomization), fixed a priori; discriminative gate: trivial copy → 1.000,
  adversarial sign-flip → 0.002.
- τ_audit = 0.161 (shortcut detection), calibrated from a known-positive ground truth.
- Seeds: [42, 1, 7, 2, 3, 4, 5, 6, 8, 9].
