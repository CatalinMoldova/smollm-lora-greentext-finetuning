# Generation Examples

Fixed prompts used before and after LoRA fine-tuning. Full before/after outputs are in
[`report/lora_smollm_greentext_ablation_report.pdf`](../report/lora_smollm_greentext_ablation_report.pdf).

## Summary

After fine-tuning, the model learns the greentext format (`>`-prefixed imperative clauses).
Semantic coherence improves over the base model, but **repetition** is the dominant failure mode
(training on only 100 examples for ~20 effective epochs).

## Prompt 1

```
> Embark on a global treasure hunt
> Decode cryptic clues
```

**Before fine-tuning:** Base SmolLM-135M produces unstructured or off-format continuations.

**After fine-tuning (r=16):** Model produces greentext-style `>` lines; see PDF for exact outputs.

**After ablation (r=4):** Similar format adoption; slightly slower convergence during training.

## Prompt 2

```
> Work as a cashier
> Customer counts their change
 > Realize the importance of patience 
 >
```

**Before fine-tuning:** Base SmolLM-135M produces unstructured or off-format continuations.

**After fine-tuning (r=16):** Model produces greentext-style `>` lines; see PDF for exact outputs.

**After ablation (r=4):** Similar format adoption; slightly slower convergence during training.

## Prompt 3

```
> Explore the ruins of Pompeii
> Uncover ancient artifacts
>
```

**Before fine-tuning:** Base SmolLM-135M produces unstructured or off-format continuations.

**After fine-tuning (r=16):** Model produces greentext-style `>` lines; see PDF for exact outputs.

**After ablation (r=4):** Similar format adoption; slightly slower convergence during training.

## Prompt 4

```
> Take on the role of a homeless person
> Experience the hardships and stigma
> Learn the true meaning of resilience
>
```

**Before fine-tuning:** Base SmolLM-135M produces unstructured or off-format continuations.

**After fine-tuning (r=16):** Model produces greentext-style `>` lines; see PDF for exact outputs.

**After ablation (r=4):** Similar format adoption; slightly slower convergence during training.

