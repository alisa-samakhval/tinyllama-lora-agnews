# tinyllama-lora-agnews

Fine-tuning [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) for four-class news topic classification (World, Sports, Business, Sci/Tech) using Low-Rank Adaptation (LoRA) with 4-bit quantization (QLoRA).

## Results

| Configuration | Accuracy | Balanced Accuracy | F1 |
|---|---|---|---|
| Base model (no fine-tuning) | 0.28 | 0.23 | 0.17 |
| LoRA r=8 | 0.85 | 0.85 | 0.85 |
| LoRA r=16 | 0.86 | 0.86 | 0.86 |

Perplexity dropped from **4,686,015** (base) to **1.34** (r=8) on the evaluation set.

Only **0.10% of parameters** are trainable (1,126,400 of 1,101,174,784). The saved adapter checkpoint is a few MB versus ~2GB for the quantized base model.

## Method

Classification is framed as text generation. The model receives a prompt and is trained to produce the correct label as its next tokens:

```
Determine the topic of the news article and respond with one word only: World, Sports, Business, or Sci/Tech.
Article: {text}
Topic:
```

At inference, the output is parsed by searching for the first label keyword after `Topic:`.

**LoRA configuration:**
- Target modules: `q_proj`, `v_proj` (standard for LLaMA-style architectures)
- Rank: r=8 and r=16 (compared)
- Alpha: 16 / 32 respectively
- Dropout: 0.05

**Training setup:**
- Base model loaded in 4-bit NF4 quantization with bfloat16 compute dtype
- 5 epochs, batch size 4, gradient accumulation steps 4 (effective batch 16)
- Learning rate: 1e-3
- Max sequence length: 512 tokens
- Training data: 1,000 samples from AG News train split
- Evaluation: next 100 samples (train[1000:1100])

## Setup

```bash
pip install torch transformers peft bitsandbytes datasets accelerate
```

Then open and run `lora_finetune.ipynb` top to bottom.

## Files

```
├── lora_finetune.ipynb   # Full training and evaluation pipeline
├── requirements.txt
├── report.pdf            # Course report (COMP 388 – LLMs, Loyola University Chicago)
└── README.md
```

## References

- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Dettmers et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [AG News dataset](https://huggingface.co/datasets/ag_news)
