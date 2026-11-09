"""DATA PREPROCESSING — prepare mixed data without leaking information.

Missing values, invalid types, units, and categorical columns can dominate the
geometry that an unsupervised model sees. Preserve raw data, document choices,
and fit learned transformations on training rows only.

IMPLEMENTATION map: (1) DataFrame creates mixed features and missing values;
(2) ColumnTransformer selects numeric versus categorical treatment; (3) the
Pipeline binds preprocessing to K-Means; (4) fit_predict learns from X only and
predict applies the same fitted transformations to a new row.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

frame = pd.DataFrame({"monthly_spend": [45, 52, np.nan, 210, 195, 230, 78, 85, 91, 320, 305, 288],
                      "visits": [2, 3, 2, 10, 9, 12, 4, np.nan, 5, 14, 13, 15],
                      "region": ["north", "north", "west", "south", "south", "south",
                                 "west", "west", "west", "north", "north", "south"]})
numeric, categorical = ["monthly_spend", "visits"], ["region"]
# IMPLEMENTATION 2: median/mode imputation, scaling and one-hot encoding.
prep = ColumnTransformer([
    ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
    ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])
# IMPLEMENTATION 3: transforms are learned and kept with the clusterer.
model = Pipeline([("preprocess", prep),
                  ("cluster", KMeans(n_clusters=3, n_init=10, random_state=7))])
labels = model.fit_predict(frame)  # IMPLEMENTATION 4: training uses features only
print("Input rows:", len(frame))
print("Transformed shape:", model.named_steps["preprocess"].transform(frame).shape)
print("Cluster labels:", labels.tolist(), "(IDs are arbitrary)")
new_row = pd.DataFrame({"monthly_spend": [100], "visits": [5], "region": ["east"]})
print("New row cluster:", int(model.predict(new_row)[0]))
