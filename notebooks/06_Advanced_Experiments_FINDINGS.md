# Phase 5: Advanced Experiments — Findings

> **Status: complete.** `06_advanced_experiments.ipynb` ran end-to-end on Colab (T4), reusing the Phase-4 `mistral_qlora_gen` adapter from Drive. Numbers below are from that run.

## Goal
With the core system done (Phases 0–4), explore four directions that make SupportAI **smarter and cheaper to operate**. All four roadmap options implemented.

## Setup
- Colab T4. Inputs from Drive: data CSV, `label_encoder.pkl`, Phase-1 `tfidf_vectorizer.pkl` + `lr_tfidf.pkl`, Phase-4 `mistral_qlora_gen/` adapter (loaded successfully).
- Same 70/15/15 split (seed 42) as all prior notebooks. Rows 26,872; 11 categories; 27 intents.

---

## A · RAG over historical responses — *the instructive negative result*
Embed all train instructions (`all-MiniLM-L6-v2`) → FAISS (cosine); retrieve top-`k=3`; ground the fine-tuned Mistral in them.

| Metric | Value |
|--------|-------|
| category-match@3 (retrieval) | **1.0000** |
| ROUGE-L retrieve-and-return (no LLM) | 0.3614 |
| ROUGE-L fine-tuned, no RAG | **0.3724** |
| ROUGE-L fine-tuned + RAG | 0.3536 |

**What happened:** retrieval was *perfect* — the dataset is full of near-duplicate paraphrases, so the nearest historical ticket almost always shares the category (the smoke test pulled three near-identical PAYMENT tickets). But grounding **lowered** ROUGE (0.372 → 0.354), and the zero-cost retrieve-and-return baseline (0.361) beat FT+RAG. On a clean, low-diversity dataset the fine-tuned model has already memorized the house style, so retrieved context mostly adds length/noise.

**Takeaway:** RAG's value here is **not** ROUGE — it's **grounding and auditability** (replies traceable to real resolutions, less hallucination), and it would pay off on **harder, more varied real traffic** where the model can't just recall the style. Consistent with the whole project's "match the method to the data" thesis.

## B · Active learning — *a real operational saving*
Entropy (uncertainty) vs random sampling on the Phase-1 TF-IDF+LogReg; learning curves in `experiments/phase5_active_learning.png`.

| Metric | Value |
|--------|-------|
| labels to reach 0.95 macro-F1 — uncertainty | **1000** |
| labels to reach 0.95 macro-F1 — random | 1200 |
| labeling-cost reduction | **16.7%** |

Uncertainty sampling hit the target with ~17% fewer labels. For a team paying humans to label tickets, prioritizing the model's least-confident cases is a direct cost saving.

## C · Prompt engineering (chain-of-thought)
The 80 hardest tickets (lowest LogReg confidence, max-proba 0.16–0.59); zero-shot vs few-shot vs CoT with Mistral.

| Strategy | Hard-subset accuracy |
|----------|----------------------|
| zero-shot | 0.8250 |
| few-shot | 0.7875 |
| chain-of-thought | **0.8500** |

Two findings: (1) **few-shot *hurt*** on hard tickets — one demo per category biased the model toward demoed phrasings on genuinely ambiguous inputs; (2) **CoT was best** (+2.5 pts over zero-shot) — reasoning-before-answering is what helps where it's hard. But all three remain far below fine-tuned DistilBERT (0.999): prompting narrows the hard-case gap, it doesn't close it.

## D · Multi-task learning — *a clean win*
Shared DistilBERT encoder + two heads (`category` 11-way, `intent` 27-way) vs single-task baselines trained identically.

| Task | Multi-task F1 | Single-task F1 |
|------|---------------|----------------|
| category | **0.9993** | 0.9986 |
| intent | 0.9966 | 0.9967 |

The shared encoder **matched (even slightly beat) separate models** — parity at ~half the parameters and serving cost. When tasks share structure, one encoder serving both predictions is strictly the better deployment.

_Scope note: "multi-task" here = the two classification tasks; a shared encoder can't emit free-text generation (needs a decoder). Joint classification+generation remains future work._

---

## PM verdict
- **Ship**: TF-IDF+LogReg routing (Phase 1); a **multi-task encoder** to serve `category`+`intent` from one model; QLoRA-Mistral for reply drafting. Keep **RAG for grounding/auditability**, not ROUGE — revisit on more diverse real traffic.
- **Operate cheaper**: **active learning** (~17% fewer labels); **CoT** as a fallback on low-confidence tickets.
- **Next (Phase 6)**: Gradio demo (ticket → category + intent + confidence + drafted reply) and the human-eval rubric from `METRICS.md` — the metric that actually judges generation, since ROUGE clearly doesn't on this data.

## Artifacts
- `experiments/phase5_results.csv` — 14 rows across the four experiments.
- `experiments/phase5_active_learning.png` — learning-curve plot.
