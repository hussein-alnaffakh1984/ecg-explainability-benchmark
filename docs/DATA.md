# Data

This project uses **PTB-XL** (a large publicly available 12-lead ECG dataset).
PTB-XL is **not** redistributed in this repository.

- Source: PhysioNet — PTB-XL v1.0.1.
- Sampling rate used: 100 Hz (records100).
- After excluding records without a superclass label, 21,430 records remain;
  patient-disjoint splits are train 17,111 / val 2,156 / test 2,163.
- `src/01_rebuild_and_train.py` rebuilds `X.npy`, `Y.npy`, `split_index.csv`
  deterministically from the raw release. Set the raw dataset path at the top of
  the script (default points to the Kaggle PTB-XL mirror).

Please obtain PTB-XL from its official source and follow its license/terms.
