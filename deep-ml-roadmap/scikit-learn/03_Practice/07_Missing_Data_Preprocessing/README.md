# Missing Data Preprocessing

**Type**: Preprocessing showcase  
**Dataset**: Synthetic (800 samples) with realistic missing patterns  
**Focus**: Explicit demonstration of the imputation pipeline

## The Pipeline

```
Missing values (8-12% per column)
    → SimpleImputer(strategy='median')       for numeric
    → SimpleImputer(strategy='most_frequent') for categorical
    → OrdinalEncoder                          for ordinal
    → OneHotEncoder                           for nominal
    → StandardScaler
    → Model
```

## Skills Demonstrated
- MCAR vs MAR missingness patterns
- All `SimpleImputer` strategies compared
- `ColumnTransformer` for mixed-type imputation
- Why dropping rows loses too much data
- Proving imputer uses only training statistics

## Run
```bash
python main.py
```
