# End-to-End ML Project — Employee Salary Prediction

**Type**: Regression  
**Dataset**: Synthetic (1500 employees) — `data/raw/employee_salary.csv`  
**Target**: `salary` (annual, in USD)

## Full ML Lifecycle

```
Step 1:  Get data (auto-generates if not found)
Step 2:  EDA — distributions, correlations, missing values
Step 3:  Cleaning — duplicates, bad rows
Step 4:  Train/Test Split (before preprocessing!)
Step 5-7: Feature Engineering + ColumnTransformer + Pipeline
Step 8:  Cross-validation across 4 models
Step 9:  RandomizedSearchCV to tune GradientBoosting
Step 10: Final evaluation on test set (once)
Step 11: joblib.dump — save full pipeline
Step 12: joblib.load → predict on new raw data
```

## Key Design Decisions

- Split before preprocessing — enforced by Pipeline
- Feature engineered `experience_to_age_ratio` before split but added consistently
- Ordinal encoding for education and performance (ordered categories)
- `loguniform` distribution for learning_rate in RandomizedSearch
- Pipeline saved includes preprocessor — no manual transformation at inference

## Run
```bash
# Generate data (if needed) and run full pipeline:
python main.py
```
