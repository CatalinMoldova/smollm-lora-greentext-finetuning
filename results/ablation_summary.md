# Ablation Summary: LoRA rank r=16 vs r=4

## Setup
- Base model: `HuggingFaceTB/SmolLM-135M`
- Dataset: `maxmyn/wholesome_greentext_110k` (first 100 rows)
- Fixed: `lora_alpha=32`, `lora_dropout=0.05`, `lr=2e-4`, batch=4, max_steps=500
- Changed: LoRA rank `r=16` (main) vs `r=4` (ablation)

## Perplexity (training set, exp(loss))

| Step | r=16 | r=4 |
|---:|---:|---:|
| 0 | 29.96 | 29.95 |
| 100 | 4.68 | 4.37 |
| 200 | 1.49 | 2.55 |
| 300 | 1.25 | 1.44 |
| 400 | 1.19 | 1.29 |
| 500 | 1.28 | 1.29 |

## Findings
1. **Both ranks reach similar final perplexity** (~1.28–1.29 at step 500).
2. **r=16 converges faster** — lower perplexity by steps 200–400.
3. **r=4 is slower but catches up** by step 500.
4. **Generation quality** is comparable; both adopt greentext format with repetition artifacts.
5. **Caveat:** α was fixed at 32 while r was halved, so the ablation is partially confounded.

## W&B runs
- Main run: `my_first_lora_run` (r=16)
- See `assets/training_loss.png` for the training loss curve.
