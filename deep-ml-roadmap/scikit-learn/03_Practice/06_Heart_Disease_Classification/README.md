# Heart Disease Classification

**Type**: Binary Classification (medical diagnosis)  
**Dataset**: Breast Cancer (sklearn built-in, used as medical proxy)  
**Target**: Disease positive/negative

## Skills Demonstrated
- `SelectKBest` with ANOVA F-statistic for feature selection
- `RFECV` for automatic optimal feature count
- `class_weight='balanced'` for imbalanced medical data
- Recall as primary metric (missing positives is costly)
- Decision threshold tuning: lower threshold → higher recall
- ROC curve + confusion matrix

## Key Insight
In medical classification, **Recall** matters more than Precision. 
A missed diagnosis (false negative) is worse than a false alarm (false positive).

## Run
```bash
python main.py
```
