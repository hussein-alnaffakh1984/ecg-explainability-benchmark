# Numeric results (values only)

10 seeds [42,1,7,2,3,4,5,6,8,9]; ABPC reported vs random.

## Classifier
Test macro-AUROC: 0.9262 ± 0.0009 (range 0.9249–0.9276).

## Axis A — faithfulness (ABPC vs random), seed-mean per class
| Method | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|
| DeepSHAP | +0.295 | +0.032 | +0.049 | -0.016 | +0.254 |
| Integrated Gradients | +0.298 | +0.032 | +0.048 | -0.017 | +0.253 |
| DeepLIFT | +0.306 | +0.039 | +0.042 | -0.017 | +0.244 |
| Grad×Input | +0.275 | +0.029 | +0.044 | -0.015 | +0.248 |
| Saliency | +0.030 | +0.012 | -0.011 | +0.026 | +0.207 |
| Grad-CAM | +0.054 | +0.011 | +0.004 | +0.005 | +0.019 |
| Random | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

MI pooled per-record (n=950): DeepLIFT +0.039 [+0.031,+0.047], IG +0.033 [+0.024,+0.041],
DeepSHAP +0.032 [+0.024,+0.040], Grad×Input +0.030 [+0.022,+0.038],
Saliency +0.012 [+0.006,+0.018], Grad-CAM +0.011 [+0.009,+0.014].

## Shortcut injection (separation AUROC = 1.00, τ_audit = 0.161)
| Method | Fraction on spike | Detection |
|---|---|---|
| DeepLIFT | 0.485 | 100% |
| Grad×Input | 0.417 | 100% |
| Integrated Gradients | 0.411 | 99% |
| DeepSHAP | 0.406 | 100% |
| Saliency | 0.052 | 0% |
| Grad-CAM | 0.010 | 0% |

## Mixed model (per-record ABPC; method×class fixed, random intercept seed)
DeepLIFT/IG/Grad×Input do not differ from DeepSHAP (all p>0.8); Grad-CAM and Saliency
carry large negative method×class interactions on high-signal classes.
