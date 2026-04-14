# Classical Baseline Model Findings

## 📊 Validation Results Summary
Three classical machine learning algorithms were evaluated on the validation set to establish a performance baseline. All models were trained on the same TF-IDF feature space.

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | Training Time |
|:------|:--------:|:----------:|:-------------:|:-------------:|
| **Logistic Regression** | `0.9960` | `0.9955` | `0.9960` | `1.82s` |
| **XGBoost** | `0.9935` | `0.9933` | `0.9935` | `41.81s` |
| **Random Forest** | `0.9804` | `0.9844` | `0.9802` | `2.18s` |

## 🔍 Key Observations

### ✅ Logistic Regression: Optimal Baseline
- Achieved the **highest performance** across all metrics (~99.6%).
- **Fastest training time** (1.82s), offering exceptional computational efficiency.
- Indicates the TF-IDF feature space is highly linearly separable for this classification task.

### ⚠️ XGBoost: High Performance, High Cost
- Delivered strong results (~99.35%), trailing Logistic Regression by <0.3%.
- Required **~23x longer training time** (41.81s) with no measurable gain.
- Suggests diminishing returns for gradient-boosted trees on this sparse, high-dimensional text representation.

### ⚠️ Random Forest: Suboptimal for Sparse Features
- Lowest performance among the three (~98.0% accuracy, ~98.4% macro F1).
- Tree-based models often underperform on high-dimensional sparse data due to inefficient feature splitting and lack of linear weight optimization.
- Training time (2.18s) is reasonable but not justified by the performance drop.

## ⚖️ Efficiency vs. Performance Trade-off
- **Best Accuracy/Speed Ratio:** Logistic Regression
- **Marginal Gain vs Compute:** XGBoost offers negligible improvement over LR at a significantly higher computational cost.
- **Dataset Characteristics:** Consistently high scores (>98%) across all models indicate a well-structured, highly separable problem. Sparse TF-IDF vectors inherently favor linear models.

## ✅ Recommendation
**Adopt Logistic Regression as the primary baseline.** It delivers top-tier validation performance with minimal computational overhead, making it ideal for rapid iteration and production deployment. XGBoost may be revisited if advanced ensemble tuning is required, but it is unnecessary for current performance targets. Random Forest is not recommended for this specific feature engineering pipeline.