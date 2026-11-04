# Trained weights

## Transformer encoder (3 seeds)
`transformer_weights.zip` in this folder contains `transformer_seed{42,1,7}.pt`
(0.665M parameters each; test macro-AUROC 0.9112, 0.9127, 0.9121).

## ResNet1D (10 seeds)
The ten ResNet1D models (8.74M parameters each, seeds [42, 1, 7, 2, 3, 4, 5, 6, 8, 9]) are too large
for the git tree. Two options:

1. **Download** `resnet1d_seed*.pt` from the Kaggle dataset
   https://www.kaggle.com/datasets/hussein1984/ecg-models (folder `models/`).
2. **Regenerate** with `python src/01_rebuild_and_train.py`; training is seed-deterministic and
   reproduces a mean test macro-AUROC of 0.9262.

Place the files in `models/` as `resnet1d_seed{42,1,7,2,3,4,5,6,8,9}.pt`.
