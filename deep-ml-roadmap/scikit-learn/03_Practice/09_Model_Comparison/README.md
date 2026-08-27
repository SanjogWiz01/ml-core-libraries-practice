# Model Comparison

**Type**: Multi-class Classification  
**Dataset**: Wine (sklearn built-in, 178 samples, 13 features, 3 classes)  
**Goal**: Systematic comparison of 10 classifiers using CV

## Models Compared
1. Gaussian Naive Bayes (fast baseline)
2. Logistic Regression
3. KNN (k=5 and k=9)
4. Decision Tree (max_depth=5)
5. SVM (linear and RBF kernels)
6. Random Forest (100 trees)
7. Extra Trees (100 trees)
8. Gradient Boosting

## Methodology
- 10-fold StratifiedKFold (higher k for small dataset)
- `cross_validate` with accuracy + F1-macro + fit time
- `return_train_score=True` to detect overfitting
- Final test evaluation on held-out set

## Skills Demonstrated
- Systematic model selection workflow
- Train vs validation score comparison (overfitting check)
- Mean ± std for stable comparison
- Visualization: accuracy bar chart + F1 mean vs stability scatter

## Run
```bash
python main.py
```
