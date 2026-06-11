# Phase 4: LLM Fine-tuning (PEFT / LoRA / QLoRA) — Findings

> **Status: scaffold.** The notebook (`05_llm_finetuning.ipynb`) is built and runnable on Colab (T4) but has **not been executed end-to-end yet**. Fill the `[ ]` placeholders from your run — the structure below is the argument the numbers should support.

## Goal of this phase
Phases 0–3 showed classification is *solved* (TF-IDF + LogReg = 0.995 macro-F1, DistilBERT ≈ 0.999). Phase 4 stops chasing classification accuracy and asks a different question: **what does it cost to bring a generative LLM to this problem, and is the new capability (writing replies) worth it?**

## What was run
- **Response generation** via supervised fine-tuning (SFT) with **QLoRA** (4-bit NF4 + LoRA adapter).
- Two model sizes: **Phi-2 (2.7B)** for cheap pipeline validation, **Mistral-7B-Instruct** for the headline result.
- **LLM classification**: zero-shot + few-shot with Mistral-Instruct, compared to the BERT baseline.
- **LoRA vs QLoRA**: measured on Phi-2 (VRAM, train time, ROUGE-L).

## Setup
- Runtime: Google Colab, T4 (16 GB).
- Subsample: `N_TRAIN_GEN=2000` generation examples, `N_EVAL=200` ROUGE pool, `N_CLS_EVAL≈330` classification tickets.
- LoRA: r=16, α=32, dropout=0.05, target=all-linear. 1 epoch, lr=2e-4, max_seq_len=512.

## Results (fill in)

### Generation — ROUGE-L (base → fine-tuned)
| Model | Base | Fine-tuned (QLoRA) | Δ |
|---|---|---|---|
| Phi-2 (2.7B) | [ ] | [ ] | [ ] |
| Mistral-7B-Instruct | [ ] | [ ] | [ ] |

### Classification — macro-F1 (LLM vs prior phases)
| Approach | macro-F1 | Notes |
|---|---|---|
| Mistral zero-shot | [ ] | no training |
| Mistral few-shot (11 demos) | [ ] | one demo/category |
| DistilBERT (Phase 3) | 0.9999 | fine-tuned, far cheaper |
| TF-IDF + LogReg (Phase 1) | 0.9955 | deployed baseline |

### Efficiency — LoRA vs QLoRA (Phi-2)
| Method | Peak VRAM (GB) | Train time (s) | ROUGE-L |
|---|---|---|---|
| QLoRA (4-bit) | [ ] | [ ] | [ ] |
| LoRA (fp16) | [ ] | [ ] | [ ] |

## Expected conclusions (validate against the numbers)
1. **Fine-tuning visibly improves generation** — base models miss the dataset's house style (placeholder tokens like `{{Order Number}}`, polite closings); fine-tuned models adopt it. ROUGE-L should rise.
2. **LLM classification < BERT** — zero/few-shot Mistral will likely land below 0.999 and is far slower. Confirms: for *this* task, fine-tuned BERT (or even LogReg) remains the right call.
3. **QLoRA ≈ free quality, big VRAM win** — 4-bit should use a fraction of fp16-LoRA's VRAM with negligible ROUGE difference, and is what makes the 7B fit a T4 at all.

## PM verdict (draft)
- **Routing/classification** → keep TF-IDF + Logistic Regression (Phase 1). Nothing in Phase 4 beats it on cost-adjusted accuracy.
- **Reply drafting** → a QLoRA-fine-tuned 7B is the first approach producing usable, on-brand replies. Justifies an LLM in the stack as an **agent assist** (human-in-the-loop), not full automation, pending the Phase 5 human-eval rubric.

## Artifacts
- `experiments/phase4_results.csv` — tidy results (generation / classification / efficiency).
- `models/phi2_qlora_gen/`, `models/mistral_qlora_gen/` — LoRA adapters (weights gitignored) + model-card `README.md`.

## Next (Phase 5)
RAG over historical responses to ground generation in real resolutions; human-eval rubric on a 100-response sample (see `METRICS.md`).
