## Phase 1 & 2: Key Insight - Model Selection - Classical Baseline

**Critical Finding**: Logistic Regression achieved 99.5% F1-Macro, leaving minimal room for improvement.

### Results Summary

| Model                      | F1 (Macro) | Parameters | Training Time (s) | Complexity |
|----------------------------|------------|------------|-------------------|------------|
| TF-IDF + Logistic Regression | 0.995500  | ~53K       | 2.000000          | Low        |
| GloVe + Logistic Regression  | 0.911152  | ~560       | 5.233711          | Low        |
| GloVe + Neural Network       | 0.942630  | 15,499     | 22.791644         | Medium     |
| LSTM (Bidirectional)         | 0.994236  | 1,133,611  | 1461.541888       | High       |
| BERT (base)                  | 0.999214  | 110M       | 400.859767  (GPU) | High       |
| DistilBERT                   | 0.999873  | 67M        | 166.463498  (GPU) | High       |

**PM Decision**: For this specific dataset, deploying anything beyond Logistic Regression would be **over-engineering**. The 0.2-0.3% improvement doesn't justify:
- 40x slower inference
- 100-1000x higher cost
- 20x more memory

**When to use what**:
- ✅ Logistic Regression: Clean, keyword-driven classification
- ✅ Deep Learning: Ambiguous queries, context needed, semantic similarity
- ✅ LLMs: Generative responses, few-shot learning, complex reasoning
- ✅ BERT/Transformers: semantic understanding needed, context matters ("not good" vs "good"), small dataset (transfer learning helps), handling ambiguity, multi-lingual requirements

## Phase 4: LLM Fine-tuning (PEFT / QLoRA) — response generation + LLM classification

_Built in `05_llm_finetuning.ipynb` (Colab T4). Numbers below are placeholders — fill from your run; see `notebooks/05_LLM_Finetuning_FINDINGS.md`._

Phase 4 shifts the question from "classify better" (already solved) to **"is a generative LLM worth its cost, and what new capability does it add?"** The new capability is **writing replies**, not classifying.

### Response generation — ROUGE-L (base → QLoRA fine-tuned)
| Model | Params | Base | Fine-tuned | Method |
|-------|--------|------|------------|--------|
| Phi-2 | 2.7B | [TBD] | [TBD] | QLoRA (4-bit) |
| Mistral-7B-Instruct | 7B | [TBD] | [TBD] | QLoRA (4-bit) |

### LLM classification vs prior phases (macro-F1)
| Approach | Macro-F1 | Cost note |
|----------|----------|-----------|
| Mistral zero-shot | [TBD] | no training, slow inference |
| Mistral few-shot | [TBD] | 11 demos in-context |
| DistilBERT (Phase 3) | 0.9999 | fine-tuned, ~100x cheaper to serve |
| TF-IDF + LogReg (Phase 1) | 0.9955 | deployed baseline |

### LoRA vs QLoRA (measured on Phi-2)
| Method | Peak VRAM | Train time | ROUGE-L |
|--------|-----------|-----------|---------|
| QLoRA (4-bit) | [TBD] | [TBD] | [TBD] |
| LoRA (fp16) | [TBD] | [TBD] | [TBD] |

**PM Decision (expected)**: keep TF-IDF + Logistic Regression for **classification/routing** — Phase 4 confirms nothing beats it on cost-adjusted accuracy. Introduce a **QLoRA-fine-tuned 7B only for reply drafting**, as a human-in-the-loop agent assist, pending the Phase 5 human-eval rubric. QLoRA (not fp16 LoRA) is what makes the 7B trainable on a 16 GB T4.