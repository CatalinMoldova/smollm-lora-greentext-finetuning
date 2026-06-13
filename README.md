# LoRA Fine-Tuning of SmolLM-135M on Greentexts

Parameter-efficient supervised fine-tuning of `HuggingFaceTB/SmolLM-135M` using Low-Rank Adaptation (LoRA), with rank ablation, Weights & Biases tracking, perplexity evaluation, and qualitative generation analysis.

## Overview

This project explores how LoRA can be used to fine-tune a small language model efficiently on limited hardware. Instead of updating all model weights, LoRA freezes the base model and trains low-rank adapter matrices inside selected attention and MLP projection layers.

The experiment fine-tunes `SmolLM-135M` on a small greentext-style dataset and compares two LoRA ranks:

* `r = 16`
* `r = 4`

The main goal is not only to reduce training loss, but to understand how rank, dataset size, and training duration affect generation quality, memorization, and repetition.

## Key Question

Can a small language model learn the surface style of a dataset through LoRA fine-tuning, and what failure modes appear when the dataset is too small?

## Why This Project Matters

LoRA is widely used for parameter-efficient fine-tuning of language models because it reduces memory cost and makes adaptation possible on consumer or free cloud GPUs. This project demonstrates the full workflow:

* Loading a pretrained Hugging Face language model
* Applying LoRA adapters with PEFT
* Running supervised fine-tuning
* Logging loss with Weights & Biases
* Evaluating perplexity across checkpoints
* Comparing LoRA ranks
* Inspecting qualitative generation failures

## Technical Stack

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* TRL / SFTTrainer
* Weights & Biases
* Google Colab T4 GPU

## Model and Dataset

| Component          | Value                                                                       |
| ------------------ | --------------------------------------------------------------------------- |
| Base model         | `HuggingFaceTB/SmolLM-135M`                                                 |
| Dataset            | `maxmyn/wholesome_greentext_110k`                                           |
| Training subset    | First 100 rows                                                              |
| Fine-tuning method | LoRA                                                                        |
| Target modules     | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`, `gate_proj` |
| Main LoRA rank     | `r = 16`                                                                    |
| Ablation rank      | `r = 4`                                                                     |
| LoRA alpha         | `32`                                                                        |
| LoRA dropout       | `0.05`                                                                      |
| Learning rate      | `2e-4`                                                                      |
| Batch size         | `4`                                                                         |
| Max steps          | `500`                                                                       |
| Precision          | `fp16`                                                                      |
| Inference sampling | `temperature = 0.7`, `top_p = 0.9`                                          |

## Results

### Training Loss

The main `r = 16` run reduced training loss from approximately `3.8` to `0.2` over 500 training steps.

The first 100 steps showed the fastest improvement, suggesting that the model quickly learned the surface structure of greentexts: short lines, imperative phrasing, and the `>` prefix.

### Perplexity by Checkpoint

| Step | PPL, r = 16 | PPL, r = 4 |
| ---: | ----------: | ---------: |
|    0 |       29.96 |      29.95 |
|  100 |        4.68 |       4.37 |
|  200 |        1.49 |       2.55 |
|  300 |        1.25 |       1.44 |
|  400 |        1.19 |       1.29 |
|  500 |        1.28 |       1.29 |

The `r = 16` model reached low perplexity faster than `r = 4`, but both runs converged to nearly the same final training-set perplexity.

This suggests that, on a very small dataset, the higher-rank adapter mostly improves fitting speed rather than final quality.

## Qualitative Findings

Fine-tuning improved the model’s ability to generate greentext-style outputs. The model became more consistent at producing short `>`-prefixed lines and stayed more on-topic for some prompts.

However, generation also revealed a key failure mode: repetition.

Examples included repeated phrases such as:

```text
> Order more
> Order more
> Order more
```

and repeated quest-like completions in the treasure-hunt prompt.

This shows that low training loss does not necessarily mean good generation quality. In this experiment, low loss mostly reflected memorization caused by the tiny training set and many effective passes over the same examples.

## Main Takeaways

1. LoRA made it possible to fine-tune a 135M-parameter model efficiently on a Colab T4 GPU.
2. Higher rank (`r = 16`) fitted the small dataset faster than lower rank (`r = 4`).
3. Final training-set perplexities were almost identical for both ranks.
4. Qualitative evaluation was more revealing than loss alone.
5. The dominant failure mode was repetition caused by overfitting.
6. A proper held-out validation split is necessary for measuring generalization.

## Limitations

This was intentionally a small-scale experiment, but it has important limitations:

* Perplexity was evaluated on the training set, not a held-out validation set.
* Only 100 examples were used for fine-tuning.
* The model saw each example many times, increasing memorization.
* The ablation kept `alpha = 32` fixed while changing rank, so the effective LoRA scaling changed.
* Only a small number of qualitative prompts were tested.

## Future Work

Planned improvements:

* Add a proper train/validation/test split.
* Scale the dataset from 100 examples to 1,000–10,000 examples.
* Sweep rank/alpha pairs while keeping `alpha / r` constant.
* Add quantitative generation metrics such as distinct-n for repetition.
* Compare LoRA with full fine-tuning and prefix tuning.
* Publish the trained adapter to Hugging Face Hub.
* Add an interactive demo for before/after generation comparison.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the LoRA adapter:

```bash
python src/train_lora.py --config configs/lora_r16.yaml
```

Evaluate perplexity:

```bash
python src/evaluate_perplexity.py --checkpoint checkpoints/r16-step-500
```

Generate sample outputs:

```bash
python src/generate_samples.py --checkpoint checkpoints/r16-step-500
```

## Repository Structure

```text
src/        Training, evaluation, and generation scripts
notebooks/  Clean research notebook
configs/    LoRA configuration files
results/    Perplexity tables and generation examples
assets/     Plots and visual summaries
report/     Full written report
```

## Project Origin

This project was originally developed as part of an Advanced AI & Machine Learning course and later refactored into a standalone ML portfolio project focused on LoRA, PEFT, ablation design, and model failure analysis.

## References

* Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models”
* Hugging Face PEFT documentation
* Hugging Face TRL SFTTrainer documentation
* HuggingFaceTB/SmolLM-135M
