# Model card — mistral_qlora_gen

- **Base model**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Task**: response generation
- **Method**: QLoRA 4-bit
- **Dataset**: Bitext Customer Support (2000 train / 200 eval subset)
- **LoRA**: r=16, alpha=32, dropout=0.05, target=all-linear
- **Train config**: epochs=1, lr=0.0002, max_seq_len=512

## Metrics
```json
{
  "method": "QLoRA 4-bit",
  "rougeL_base": 0.2668,
  "rougeL_ft": 0.3576,
  "peak_vram_gb": 6.4,
  "train_time_s": 2238.6
}
```

## How to load
```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
model = AutoPeftModelForCausalLM.from_pretrained("models/mistral_qlora_gen", device_map="auto")
tok = AutoTokenizer.from_pretrained("models/mistral_qlora_gen")
```

## Intended use
Drafting customer-support replies for agent review (assist, not full automation).
Not validated for autonomous deployment — see ../METRICS.md for the human-eval rubric.
