## Phase 1 & 2: Key Insight - Model Selection - Classical Baseline

**Critical Finding**: Logistic Regression achieved 99.5% F1-Macro, leaving minimal room for improvement.

### Results Summary

| Model                      | F1 (Macro) | Parameters | Training Time (s) | Complexity |
|----------------------------|------------|------------|-------------------|------------|
| TF-IDF + Logistic Regression | 0.995500  | ~53K       | 2.000000          | Low        |
| GloVe + Logistic Regression  | 0.911152  | ~560       | 5.233711          | Low        |
| GloVe + Neural Network       | 0.942630  | 15,499     | 22.791644         | Medium     |
| LSTM (Bidirectional)         | 0.994236  | 1,133,611  | 1461.541888       | High       |

**PM Decision**: For this specific dataset, deploying anything beyond Logistic Regression would be **over-engineering**. The 0.2-0.3% improvement doesn't justify:
- 40x slower inference
- 100-1000x higher cost
- 20x more memory

**When to use what**:
- ✅ Logistic Regression: Clean, keyword-driven classification
- ✅ Deep Learning: Ambiguous queries, context needed, semantic similarity
- ✅ LLMs: Generative responses, few-shot learning, complex reasoning