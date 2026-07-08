# SupportAI: Success Metrics

## Business Metrics
- **Response Time Reduction**: Target 40% reduction in average ticket response time
- **Agent Productivity**: Target 30% more tickets handled per agent per day
- **Customer Satisfaction**: Maintain or improve CSAT score (>4.0/5.0)

## Technical Metrics

### Classification Task
- **Primary Metric**: Macro F1-Score (treats all categories equally)
  - Target: >0.85 for production
  - **Achieved**: ✅ **exceeded by the classical baseline alone** — TF-IDF + Logistic Regression **0.9955** (Phase 1); DistilBERT **0.9999** (Phase 3). Deeper models add <0.5% for 100–1000× the cost.
  - LLM classification is *not* competitive here: Mistral-7B tops out at **0.80 macro-F1** few-shot (Phase 4), ~0.20 below BERT.
- **Intent classification** (finer 27-way label, added Phase 5): multi-task DistilBERT **0.9966** macro-F1 — served from the *same* encoder as `category` at ~half the cost.
- **Secondary Metrics**:
  - Accuracy: Overall correct predictions
  - Per-category F1: Identify weak categories
  - Confusion Matrix: Understand misclassification patterns

### Response Generation Task
- **Automated Metrics**:
  - ROUGE-L: Overlap with reference responses — **achieved 0.358** (Mistral-7B QLoRA fine-tuned, Phase 4). RAG did **not** improve it (0.354, Phase 5): the dataset is too near-duplicate for retrieval to add signal, so **ROUGE proved a weak proxy on this data**.
  - BLEU / Perplexity: not computed (ROUGE-L used as the single automated proxy).
- **Human Evaluation** (sample 100 responses): ⬜ **pending Phase 6** — the metric that will actually arbitrate generation quality, since ROUGE clearly under-measures it here.
  - Relevance: Does it address the query? (1-5 scale)
  - Accuracy: Is information correct? (1-5 scale)
  - Tone: Appropriate customer service tone? (1-5 scale)
  - Completeness: Answers all parts? (1-5 scale)

## Experiment Comparison Framework

Filled from Phases 1–5 (detailed train-time / param counts in `EXPERIMENTS.md`). "—" = not measured for that model.

### Classification (macro-F1)
| Model | Phase | Macro-F1 | Accuracy | Memory / size | Notes |
|-------|-------|----------|----------|---------------|-------|
| **TF-IDF + Logistic Regression** | 1 | **0.9955** | 0.9960 | ~KBs, CPU | **deployed baseline** |
| XGBoost | 1 | 0.9933 | 0.9935 | small | 23× slower to train, no gain |
| Bidirectional LSTM | 2 | 0.9942 | — | 1.1M params | high cost, no gain |
| BERT-base | 3 | 0.9992 | — | 110M params, GPU | over-engineered here |
| DistilBERT | 3 | **0.9999** | — | 67M params, GPU | best classifier |
| Multi-task DistilBERT (category) | 5 | 0.9993 | — | 67M shared | + intent head, ~½ serving cost |
| Multi-task DistilBERT (intent, 27-way) | 5 | 0.9966 | — | 67M shared | finer label |
| Mistral-7B zero-shot | 4 | 0.6875 | 0.7182 | 6.4 GB (4-bit) | LLM classify — not worth it |
| Mistral-7B few-shot | 4 | 0.8002 | 0.8121 | 6.4 GB (4-bit) | still < BERT |

### Response generation (ROUGE-L)
| Approach | Phase | ROUGE-L | Memory | Notes |
|----------|-------|---------|--------|-------|
| Mistral-7B base (zero-shot) | 4 | 0.267 | 6.4 GB (4-bit) | before fine-tuning |
| **Mistral-7B QLoRA fine-tuned** | 4 | **0.358** | 6.4 GB (4-bit) | on-brand replies |
| Fine-tuned + RAG (top-3) | 5 | 0.354 | 6.4 GB + FAISS | RAG didn't help ROUGE on this data |
| Retrieve-and-return (no LLM) | 5 | 0.361 | FAISS only | zero-cost floor |
| Human eval (relevance/accuracy/tone/completeness) | 6 | ⬜ pending | — | the metric that actually decides |

## Deployment Constraints
_(targets; latency/cost not yet formally benchmarked — Phase 6)_
- **Latency**: <500ms per ticket (95th pct) — ✅ classifiers (LogReg sub-ms; DistilBERT ~ms on GPU) meet this easily. ⚠️ The 7B reply-drafting LLM is far slower and is scoped as an **async, human-in-the-loop assist**, not inline routing.
- **Memory**: <8GB GPU for inference — ✅ Mistral-7B in 4-bit QLoRA peaked **~6.4 GB**; all classifiers are CPU-friendly.
- **Cost**: <$0.01 per ticket — ✅ classical routing is effectively free; ⚠️ LLM drafting cost is tracked separately and justified only as an agent assist.