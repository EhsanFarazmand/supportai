# SupportAI — Intelligent Ticket Routing & Response System

A learning/research project benchmarking the full spectrum of NLP approaches — from classical ML to transformers and (planned) LLM fine-tuning — for **customer-support ticket classification and response generation**.

The project is framed as an AI/PM portfolio study: the goal isn't the highest possible score, but understanding the **cost/performance tradeoff** between approaches and making a defensible deployment decision.

## Headline finding

On the [Bitext Customer Support Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), **TF-IDF + Logistic Regression reaches 0.995 macro-F1** — and fine-tuned BERT/DistilBERT only improve to ~0.999. For this clean, keyword-driven dataset, deploying a transformer is **over-engineering**: it costs ~40× slower inference and 100–1000× more compute for a 0.2–0.4% gain.

| Model | Macro-F1 | Parameters | Train time | Complexity |
|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | **0.9955** | ~53K | 1.8 s | Low |
| GloVe + Logistic Regression | 0.9112 | ~560 | 5.2 s | Low |
| GloVe + Neural Network | 0.9426 | 15.5K | 22.8 s | Medium |
| Bidirectional LSTM | 0.9942 | 1.1M | 1462 s | High |
| BERT (base) | 0.9992 | 110M | 401 s (GPU) | High |
| DistilBERT | 0.9999 | 67M | 166 s (GPU) | High |

**Phase 4 (LLM fine-tuning)** confirms and extends this: an off-the-shelf Mistral-7B classifies at only **0.80 macro-F1 few-shot** (vs DistilBERT's 0.999), so the LLM stays off the classification path. Its real value is the *new* capability — **response generation**: QLoRA fine-tuning lifts ROUGE-L to 0.30 (Phi-2) / **0.36 (Mistral-7B)**, the first models that draft usable, on-brand replies. See [05_LLM_Finetuning_FINDINGS.md](notebooks/05_LLM_Finetuning_FINDINGS.md).

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full analysis and [METRICS.md](METRICS.md) for success criteria.

## Dataset

- **Source**: Bitext Customer Support Dataset (downloaded via HuggingFace `datasets`)
- **Size**: 26,872 query/response pairs
- **Categories** (11): ACCOUNT, ORDER, REFUND, INVOICE, CONTACT, PAYMENT, FEEDBACK, DELIVERY, SHIPPING, SUBSCRIPTION, CANCEL
- **Characteristics**: queries are short (~8.7 words avg), responses are long and highly variable (~105 words avg, up to 2,472 chars)

## Project structure

```
supportai/
├── data/          # processed data + cached train/val/test splits (raw data gitignored)
├── notebooks/     # one numbered notebook per phase, each with a *_FINDINGS.md
├── models/        # saved encoders, vectorizers, and fine-tuned models (weights gitignored)
├── experiments/   # results CSVs and comparison plots
├── src/           # reusable code        (planned)
└── app/           # demo interface       (planned)
```

## Notebooks (run in order)

The notebooks form a dependency chain — each consumes artifacts produced by the previous one.

| Notebook | Phase | What it does |
|---|---|---|
| [`01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) | 0 — Foundation | Downloads the dataset, runs EDA (category distribution, ticket length, class imbalance), writes `data/customer_support_clean.csv` |
| [`02_classical_baseline.ipynb`](notebooks/02_classical_baseline.ipynb) | 1 — Classical ML | TF-IDF + Logistic Regression / Random Forest / XGBoost; cross-validation; error analysis |
| [`03_embeddings_and_neural_nets.ipynb`](notebooks/03_embeddings_and_neural_nets.ipynb) | 2 — Deep learning | GloVe embeddings, a feed-forward net, and a BiLSTM; t-SNE visualization |
| [`04_transformers.ipynb`](notebooks/04_transformers.ipynb) | 3 — Transformers | Fine-tunes BERT and DistilBERT via the HuggingFace Trainer API |
| [`05_llm_finetuning.ipynb`](notebooks/05_llm_finetuning.ipynb) | 4 — LLM fine-tuning | QLoRA fine-tunes Phi-2 + Mistral-7B for **response generation**; LLM-vs-BERT classification; LoRA-vs-QLoRA efficiency. **Runs on Colab (T4)** |
| [`06_advanced_experiments.ipynb`](notebooks/06_advanced_experiments.ipynb) | 5 — Advanced | **RAG** over historical responses, **active learning**, **chain-of-thought** prompting, and **multi-task** (category+intent) learning. **Runs on Colab (T4)** |

Each notebook has a matching `*_FINDINGS.md` documenting its results.

## Roadmap

| Phase | Status |
|---|---|
| 0 — Foundation (EDA, metrics) | ✅ Complete |
| 1 — Classical ML baseline | ✅ Complete |
| 2 — Deep learning / embeddings | ✅ Complete |
| 3 — Transformer classification | ✅ Complete |
| 4 — LLM fine-tuning (LoRA/QLoRA, response generation) | ✅ Complete |
| 5 — Advanced experiments (RAG, active learning, CoT, multi-task) | ✅ Complete |
| 6 — Product & demo (Gradio/Streamlit + PM docs) | ⬜ Planned |

## Getting started

> **Note**: `requirements.txt` is currently empty — there is no pinned dependency list yet.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies (until requirements.txt is populated)
pip install pandas numpy scikit-learn xgboost joblib datasets transformers torch matplotlib

# 3. Launch Jupyter from the notebooks/ directory and run notebooks 01 → 04 in order
cd notebooks
jupyter notebook
```

Notes:
- **Notebooks 01–03** expect the working directory to be `notebooks/` (they use `../data/...` and `../models/...` paths).
- **Notebook 04** was run on Colab/Kaggle and uses flat paths; fine-tuning needs a CUDA GPU for reasonable training times.
- A fresh clone has no raw data or model weights (both gitignored) — rerun notebook 01 to regenerate the cleaned dataset.

## License

See [LICENSE](LICENSE).
