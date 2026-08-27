# Scikit-learn 80/20 Learning Repository

**20% of the most important Scikit-learn concepts → 80% of practical ML capability.**

Built for someone who already knows Python, NumPy, Pandas, and Matplotlib.
The goal: go from raw data → preprocessing → model → evaluation → tuning → saving → prediction.

---

## Technologies

| Tool | Role |
|---|---|
| `scikit-learn` | Core ML library |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib / seaborn` | Visualization |
| `joblib` | Model persistence |
| `jupyter` | Interactive exploration |

---

## Concepts Covered

**Core (highest priority)**
1. Train/test split
2. Missing value imputation (`SimpleImputer`)
3. Feature scaling (`StandardScaler`, `MinMaxScaler`, `RobustScaler`)
4. Categorical encoding (`OneHotEncoder`, `OrdinalEncoder`)
5. `Pipeline` — prevents data leakage
6. `ColumnTransformer` — mixed-type preprocessing
7. Linear regression, Ridge, Lasso, ElasticNet
8. Logistic Regression, KNN, SVM
9. Decision Trees, Random Forest, Gradient Boosting
10. Regression metrics: MAE, MSE, RMSE, R²
11. Classification metrics: accuracy, precision, recall, F1, ROC-AUC
12. Cross-validation: `KFold`, `StratifiedKFold`, `cross_val_score`
13. Hyperparameter tuning: `GridSearchCV`, `RandomizedSearchCV`

**Secondary**
14. Feature selection: `SelectKBest`, `RFE`
15. K-Means clustering + silhouette score
16. DBSCAN
17. PCA + explained variance
18. Model saving/loading with `joblib`

**Supporting theory**
19. Overfitting, underfitting, bias-variance tradeoff
20. Data leakage and how Pipelines prevent it

---

## Repository Structure

```
scikit-learn/
├── 01_Theory/               # Concept READMEs — read before coding
│   ├── 01_ML_Fundamentals/
│   ├── 02_Data_Preprocessing/
│   ├── 03_Supervised_Learning/
│   ├── 04_Model_Evaluation_Optimization/
│   ├── 05_Unsupervised_Learning/
│   └── 06_End_to_End_Workflow/
│
├── 02_Workbase/             # Executable .py implementations per topic
│   ├── 01_Regression/
│   ├── 02_Classification/
│   ├── 03_Preprocessing/
│   ├── 04_Pipelines/
│   ├── 05_Trees_Ensembles/
│   ├── 06_Model_Evaluation/
│   ├── 07_Cross_Validation/
│   ├── 08_Hyperparameter_Tuning/
│   ├── 09_Feature_Selection/
│   ├── 10_Clustering/
│   └── 11_PCA/
│
├── 03_Practice/             # End-to-end real projects
│   ├── 01_Student_Performance/
│   ├── 02_House_Price_Prediction/
│   ├── 03_Customer_Churn/
│   ├── 04_Customer_Segmentation/
│   ├── 05_Iris_Classification/
│   ├── 06_Heart_Disease_Classification/
│   ├── 07_Missing_Data_Preprocessing/
│   ├── 08_End_to_End_ML_Project/
│   └── 09_Model_Comparison/
│
├── data/
│   ├── raw/                 # Source datasets + generation script
│   └── processed/           # Cleaned/transformed datasets
│
├── models/                  # Saved .joblib model artifacts
├── requirements.txt
└── .gitignore
```

---

## Learning Roadmap

Follow this sequence:

```
01_Theory/01_ML_Fundamentals        → understand the landscape
02_Workbase/03_Preprocessing        → handle messy real data
02_Workbase/04_Pipelines            → wire preprocessing safely
02_Workbase/01_Regression           → first models
02_Workbase/02_Classification       → classification fundamentals
02_Workbase/06_Model_Evaluation     → know when your model is good
02_Workbase/07_Cross_Validation     → honest performance estimates
02_Workbase/08_Hyperparameter_Tuning → squeeze out improvement
02_Workbase/05_Trees_Ensembles      → most powerful classical models
02_Workbase/09_Feature_Selection    → work smarter, not harder
02_Workbase/10_Clustering           → unsupervised patterns
02_Workbase/11_PCA                  → dimensionality reduction
03_Practice/08_End_to_End_ML_Project → tie it all together
```

---

## Practice Projects

| # | Project | Key Skills |
|---|---|---|
| 1 | Student Performance | regression, preprocessing, evaluation |
| 2 | House Price Prediction | Pipeline, ColumnTransformer, tuning |
| 3 | Customer Churn | classification, encoding, ROC-AUC |
| 4 | Customer Segmentation | K-Means, DBSCAN, PCA |
| 5 | Iris Classification | multi-model comparison |
| 6 | Heart Disease | feature selection, class imbalance |
| 7 | Missing Data | SimpleImputer → Pipeline → Model |
| 8 | End-to-End ML | full ML lifecycle with joblib |
| 9 | Model Comparison | CV-based multi-model benchmarking |

---

## Installation

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

---

## How to Run

Every script in `02_Workbase/` and `03_Practice/` is standalone:

```bash
# from the scikit-learn/ root
python 02_Workbase/01_Regression/linear_regression.py
python 03_Practice/08_End_to_End_ML_Project/main.py
```

Theory is in Markdown — open in any editor or GitHub.

For the end-to-end project, generate the dataset first:

```bash
python data/raw/generate_datasets.py
```
