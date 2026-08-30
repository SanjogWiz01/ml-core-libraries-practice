# Data

The workbase examples mostly generate synthetic data so every `.py` file is self-contained.

For real projects, replace the synthetic arrays with your train/validation/test datasets.

Recommended structure:
- `train.csv`
- `validation.csv`
- `test.csv`

Never fit normalization/encoding on the test set.
