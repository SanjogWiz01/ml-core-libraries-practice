# utils — Shared Helpers

Reusable code shared across all library folders. Import from here; don't copy-paste.

## Structure

```
utils/
  data_loaders.py       # load common datasets (MNIST, CIFAR, CSV)
  metrics.py            # custom evaluation metrics
  plotting.py           # loss curves, confusion matrix, feature importance
  preprocessing.py      # transforms used across multiple stages
  experiment.py         # seed setting, reproducibility helpers
```

## Import Convention

Add the repo root to your Python path and import directly:

```python
from utils.plotting import plot_confusion_matrix
from utils.metrics import f1_macro
from utils.data_loaders import load_csv_split
```

## Rule

If you write a helper twice, it belongs here.
Keep functions small, documented with a one-line docstring, and dependency-light.
