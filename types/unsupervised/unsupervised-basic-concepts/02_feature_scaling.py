"""FEATURE SCALING — make numeric units comparable where geometry needs it.

Distance and variance methods can be dominated by large-unit features. Standard
scaling subtracts the training mean and divides by its standard deviation;
MinMax maps a range; RobustScaler uses median and interquartile range. Scaling is
not automatic: binary flags, sparse counts, and meaningful thresholds may need
different treatment. Scaling also does not remove outliers.

IMPLEMENTATION: data has one large outlier. Each scaler is fit and transformed;
compare ordinary rows with the outlier. With train/test data use fit_transform
only on train and transform on held-out rows.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

X = np.array([[20, .10], [25, .12], [30, .15], [35, .17],
              [40, .20], [45, .22], [50, .25], [400, .27]], dtype=float)
print("Raw feature standard deviations:", np.round(X.std(axis=0), 2))
for name, scaler in [("standard", StandardScaler()), ("min-max", MinMaxScaler()),
                     ("robust", RobustScaler())]:
    Z = scaler.fit_transform(X)  # IMPLEMENTATION: learn then apply transformation
    print(f"{name:8s} first row={np.round(Z[0], 3)}, outlier={np.round(Z[-1], 3)}")
print("Robust scaling limits outlier influence on scale; it does not remove it.")
