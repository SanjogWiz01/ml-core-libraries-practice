# Customer Churn Classification

**Type**: Binary Classification  
**Dataset**: Synthetic (2000 customers) — `data/raw/customer_churn.csv`  
**Target**: `churn` (0=stays, 1=churns)

## Skills Demonstrated
- `OneHotEncoder` for contract type, payment method, internet service
- `class_weight='balanced'` for imbalanced classes
- LogisticRegression, RandomForest, GradientBoosting comparison
- ROC-AUC as primary metric
- Decision threshold tuning (precision/recall tradeoff)
- Confusion matrix + ROC curve visualization

## Run
```bash
python main.py
```
