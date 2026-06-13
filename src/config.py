"""Shared constants for LoRA SFT on greentexts."""

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
DATASET_NAME = "maxmyn/wholesome_greentext_110k"
DATASET_SPLIT = "train[:100]"

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj", "gate_proj",
]

BASELINE_PROMPTS = [
    "> Embark on a global treasure hunt\n> Decode cryptic clues\n",
    "> Work as a cashier\n> Customer counts their change\n > Realize the importance of patience \n >",
    "> Explore the ruins of Pompeii\n> Uncover ancient artifacts\n>\n",
    "> Take on the role of a homeless person\n> Experience the hardships and stigma\n> Learn the true meaning of resilience\n>\n",
]

CHECKPOINTS = [0, 100, 200, 300, 400, 500]
WANDB_RUN_R16 = "cb5330-new-york-university/huggingface/my_first_lora_run"
