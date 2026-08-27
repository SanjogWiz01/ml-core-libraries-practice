# Student Performance Prediction

**Type**: Regression  
**Dataset**: Synthetic (1000 students) — `data/raw/student_performance.csv`  
**Target**: `final_score` (continuous, 0–100)

## Skills Demonstrated
- `SimpleImputer` (median) for numeric missing values
- `OrdinalEncoder` for `parent_education` (has order)
- `OneHotEncoder` for `gender` (nominal)
- `ColumnTransformer` for mixed types
- Multiple regression models: LinearRegression, Ridge, RandomForest, GradientBoosting
- Cross-validation with RMSE and R²

## Run
```bash
python main.py
```
