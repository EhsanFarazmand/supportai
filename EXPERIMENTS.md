## Phase 1: Key Insight - Model Selection - Classical Baseline

**Critical Finding**: Logistic Regression achieved 99.5% F1-Macro, leaving minimal room for improvement.

### Business Impact Analysis

| Model | F1-Macro | Inference Time | Cost/1000 | Memory |
|-------|----------|----------------|-----------|---------|
| Logistic Regression | 99.5% | 5ms | $0.001 | 100MB |
| BERT (projected) | ~99.7% | 200ms | $0.10 | 2GB |
| Fine-tuned LLM (projected) | ~99.8% | 500ms | $1.00 | 8GB |

**PM Decision**: For this specific dataset, deploying anything beyond Logistic Regression would be **over-engineering**. The 0.2-0.3% improvement doesn't justify:
- 40x slower inference
- 100-1000x higher cost
- 20x more memory

**When to use what**:
- ✅ Logistic Regression: Clean, keyword-driven classification
- ✅ Deep Learning: Ambiguous queries, context needed, semantic similarity
- ✅ LLMs: Generative responses, few-shot learning, complex reasoning