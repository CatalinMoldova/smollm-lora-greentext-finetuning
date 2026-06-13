"""LoRA SFT training helpers."""

from __future__ import annotations

from pathlib import Path

import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.config import DATASET_NAME, DATASET_SPLIT, LORA_TARGET_MODULES, MODEL_NAME


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_dataset_split(name: str = DATASET_NAME, split: str = DATASET_SPLIT):
    return load_dataset(name, split=split)


def build_lora_config(cfg: dict) -> LoraConfig:
    lora = cfg["lora"]
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        target_modules=lora["target_modules"],
        lora_dropout=lora["lora_dropout"],
        bias=lora.get("bias", "none"),
        task_type=lora.get("task_type", "CAUSAL_LM"),
    )


def build_training_args(cfg: dict) -> SFTConfig:
    t = cfg["training"]
    return SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        save_steps=t["save_steps"],
        learning_rate=t["learning_rate"],
        dataset_text_field=t["dataset_text_field"],
        logging_steps=t["logging_steps"],
        run_name=t["run_name"],
        max_steps=t["max_steps"],
        report_to=t.get("report_to", "wandb"),
        fp16=t.get("fp16", True),
        bf16=t.get("bf16", False),
    )


def train_from_config(config_path: str | Path, train_dataset=None):
    cfg = load_config(config_path)
    model, _ = load_model_and_tokenizer(cfg.get("model_name", MODEL_NAME))
    model = get_peft_model(model, build_lora_config(cfg))
    if train_dataset is None:
        train_dataset = load_dataset_split(cfg["dataset_name"], cfg["dataset_split"])
    trainer = SFTTrainer(
        model=model,
        args=build_training_args(cfg),
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
    )
    trainer.train()
    return trainer
