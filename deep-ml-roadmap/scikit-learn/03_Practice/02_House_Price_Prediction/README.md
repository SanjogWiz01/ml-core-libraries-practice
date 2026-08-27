# House Price Prediction

**Type**: Regression  
**Dataset**: California Housing (sklearn built-in)  
**Target**: Median house value in $100k units

## Skills Demonstrated
- Feature engineering (derived features: rooms per household, etc.)
- `ColumnTransformer` with `StandardScaler`
- 4 regression models compared via CV
- `RandomizedSearchCV` for tuning `GradientBoosting`
- Final test evaluation (once, at the end)
- `joblib.dump` / `joblib.load` for model persistence

## Run
```bash
python main.py
```
