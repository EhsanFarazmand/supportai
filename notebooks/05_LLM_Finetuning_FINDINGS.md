# Phase 4: LLM Fine-tuning (PEFT / LoRA / QLoRA) — Findings

> **Status: complete.** `05_llm_finetuning.ipynb` ran end-to-end on Colab (T4, torch 2.11 / bitsandbytes ≥0.44). Numbers below are from that run.

## Goal of this phase
Phases 0–3 showed classification is *solved* (TF-IDF + LogReg = 0.995 macro-F1, DistilBERT ≈ 0.999). Phase 4 stops chasing classification accuracy and asks a different question: **what does it cost to bring a generative LLM to this problem, and is the new capability (writing replies) worth it?**

## What was run
- **Response generation** via SFT with **QLoRA** (4-bit NF4 + LoRA adapter).
- Two model sizes: **Phi-2 (2.7B)** for pipeline validation, **Mistral-7B-Instruct-v0.3** for the headline result.
- **LLM classification**: zero-shot + few-shot with Mistral-Instruct, compared to the BERT baseline.
- **LoRA vs QLoRA**: measured on Phi-2 (VRAM, train time, ROUGE-L).

## Setup
- Runtime: Google Colab, T4 (15.6 GB), torch 2.11 / cu128.
- Models: `microsoft/phi-2` (2.7B) and `mistralai/Mistral-7B-Instruct-v0.3` (7B; extended 32768 vocab, v3 tokenizer).
- Subsample: 2000 generation train examples; ROUGE scored on **80** held-out tickets; classification on **~330** test tickets (~30/category).
- LoRA: r=16, α=32, dropout=0.05, target=all-linear. 1 epoch, lr=2e-4, max_seq_len=512, effective batch 16.

## Results

### Generation — ROUGE-L (base → QLoRA fine-tuned)
| Model | Base | Fine-tuned | Δ | Peak VRAM | Train time |
|-------|------|-----------|------|-----------|------------|
| Phi-2 (2.7B) | 0.1966 | **0.3031** | +0.1066 | 3.89 GB | 995 s |
| Mistral-7B-Instruct | 0.2668 | **0.3576** | +0.0908 | 6.40 GB | 2239 s |

Fine-tuning helped **both** models substantially (+0.09 to +0.11 ROUGE-L). Mistral-7B fine-tuned is best (0.3576); notably, **fine-tuned Phi-2 (0.3031) already beats *base* Mistral-7B (0.2668)** — a small fine-tuned model outperforms a 2.6× larger un-adapted one on this narrow task.

### Classification — macro-F1 (LLM vs prior phases)
| Approach | Accuracy | Macro-F1 | Cost note |
|----------|----------|----------|-----------|
| Mistral zero-shot | 0.7182 | 0.6875 | no training, slow inference |
| Mistral few-shot (11 demos) | 0.8121 | 0.8002 | one demo/category in-context |
| DistilBERT (Phase 3) | — | **0.9999** | fine-tuned, ~100× cheaper to serve |
| TF-IDF + LogReg (Phase 1) | — | 0.9955 | deployed baseline |

Few-shot helped a lot (+0.11 macro-F1 over zero-shot), but even few-shot (0.80) lands **~0.20 below the fine-tuned BERT** — and an order of magnitude slower/costlier per ticket. The LLM is the wrong tool for *this* classification problem.

### Efficiency — LoRA vs QLoRA (Phi-2, same data/config)
| Method | Peak VRAM | Train time | ROUGE-L |
|--------|-----------|-----------|---------|
| QLoRA (4-bit) | **3.89 GB** | 995 s | 0.3031 |
| LoRA (fp16) | 6.67 GB | **786 s** | 0.3149 |

**The nuance worth keeping:** QLoRA used **58% of the VRAM** (a 42% saving) for a **negligible −0.011 ROUGE-L** difference — confirming 4-bit quantization is nearly free in *quality*. But QLoRA was **~27% slower** (995 s vs 786 s): the 4-bit dequantization adds compute, so on a small model where memory isn't the binding constraint, fp16 LoRA is actually faster. QLoRA's value is **memory, not speed** — and memory is exactly what's binding for the 7B (Mistral QLoRA peaked at 6.40 GB; fp16 would not have fit the T4 alongside optimizer state).

## Interpreting the ROUGE numbers
Absolute ROUGE-L (~0.30–0.36) looks low because the dataset's reference replies are long and flowery ("How invigorating it is to witness your proactive approach…"), while the fine-tuned models produce **shorter, on-task** replies that correctly adopt the house style — placeholder tokens like `{{Account Type}}`, polite openings/closings. ROUGE penalizes the brevity even when the reply is good (see qualitative samples in the notebook). **ROUGE is a rough proxy here; the Phase 5 human-eval rubric matters more.**

## PM verdict
- **Routing/classification** → keep **TF-IDF + Logistic Regression** (Phase 1). Phase 4 confirms nothing beats it on cost-adjusted accuracy; even few-shot Mistral trails BERT by ~0.20 F1 at far higher cost.
- **Reply drafting** → a **QLoRA-fine-tuned 7B** is the first approach producing usable, on-brand replies. It justifies an LLM in the stack as an **agent assist** (human-in-the-loop draft), not full automation, pending human eval.
- **QLoRA over fp16 LoRA** for anything 7B+: the VRAM saving is what makes it trainable on commodity (T4) hardware; the small speed penalty and negligible quality cost are easy trades.

## Artifacts
- `experiments/phase4_results.csv` — tidy results (generation / classification / efficiency), 14 rows.
- `models/phi2_qlora_gen/`, `models/mistral_qlora_gen/` — LoRA adapters (weights gitignored) + model-card `README.md`.

## Next (Phase 5)
RAG over historical responses to ground generation in real resolutions; human-eval rubric on a 100-response sample (relevance / accuracy / tone / completeness, see `METRICS.md`) — the metric that actually arbitrates generation quality.
