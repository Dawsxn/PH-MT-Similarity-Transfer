"""
LoRA training script for mBART Seq2Seq models (PEFT)

This script implements a LoRA-based fine-tuning flow using the `peft` library
and Hugging Face `transformers` Seq2SeqTrainer. It mirrors the training
setup used in the notebook `notebooks/train-models.ipynb` but applies LoRA
adapters to the model so only the low-rank adapters are trained.

Usage example:
    python scripts/train_with_lora.py \
        --pair en-tl \
        --model facebook/mbart-large-50 \
        --output_dir models/tl_lora_test \
        --num_train_epochs 1 \
        --per_device_train_batch_size 8

Required packages (install if needed):
    pip install transformers datasets evaluate peft accelerate torch sentencepiece

Contract (inputs/outputs):
- Inputs: path to data splits in `data/splits/<pair>` containing train/dev files
  named like `train.en`, `train.tl`, `dev.en`, `dev.tl` depending on the pair.
- Outputs: saved adapter and tokenizer in the specified `output_dir`.
- Error modes: file-not-found for data, missing package import (peft), OOM on small GPUs.

Edge cases considered:
- Missing data files -> clear error message
- Missing `peft` -> instruct to install
- If GPU not available, training runs on CPU but will be slow

"""

from pathlib import Path
import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    MBartForConditionalGeneration,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# PEFT imports
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
except Exception as e:
    LORA_IMPORT_ERROR = e
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_int8_training = None
else:
    LORA_IMPORT_ERROR = None


# Default project-relative paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "splits"
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_data_for_pair(pair_name: str) -> DatasetDict:
    src_code, tgt_code = pair_name.split("-")
    pair_dir = DATA_DIR / pair_name
    if not pair_dir.exists():
        raise FileNotFoundError(f"Pair directory not found: {pair_dir}")

    def read_file(path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    train_src = read_file(pair_dir / f"train.{src_code}")
    train_tgt = read_file(pair_dir / f"train.{tgt_code}")
    dev_src = read_file(pair_dir / f"dev.{src_code}")
    dev_tgt = read_file(pair_dir / f"dev.{tgt_code}")

    train_dataset = Dataset.from_dict({"src": train_src, "tgt": train_tgt})
    dev_dataset = Dataset.from_dict({"src": dev_src, "tgt": dev_tgt})

    return DatasetDict({"train": train_dataset, "validation": dev_dataset})


def create_preprocess_fn(tokenizer: MBartTokenizerFast, src_lang: str, tgt_lang: str, max_length: int):
    def preprocess(batch: Dict):
        tokenizer.src_lang = src_lang
        enc = tokenizer(
            batch["src"], truncation=True, padding="max_length", max_length=max_length
        )
        tokenizer.tgt_lang = tgt_lang
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["tgt"], truncation=True, padding="max_length", max_length=max_length
            )
        enc["labels"] = labels["input_ids"]
        return enc

    return preprocess


def create_compute_metrics(tokenizer: MBartTokenizerFast):
    import evaluate

    bleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        return {"bleu": result["score"]}

    return compute_metrics


def apply_lora_to_model(model, r=8, alpha=16, dropout=0.1, target_modules=None, bias="none"):
    if get_peft_model is None:
        raise ImportError(
            "peft is not installed or failed to import. Install with `pip install peft`.")

    # If user passed None for target_modules, use common attention/query/key/value for mBART
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias=bias,
        task_type="SEQ_2_SEQ_LM",
    )

    # If model is in 8-bit mode, we'd prepare it differently; here we assume standard FP32/FP16
    peft_model = get_peft_model(model, lora_config)
    return peft_model


def train_pair_with_lora(
    pair_name: str,
    model_name_or_path: str,
    output_dir: Path,
    src_lang_token: str,
    tgt_lang_token: str,
    max_length: int = 128,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 3e-5,
    num_train_epochs: int = 3,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    save_total_limit: int = 2,
    fp16: bool = True,
):
    print(f"Loading tokenizer and model: {model_name_or_path}")
    tokenizer = MBartTokenizerFast.from_pretrained(model_name_or_path)
    model = MBartForConditionalGeneration.from_pretrained(model_name_or_path)

    # Apply LoRA adapters
    print("Applying LoRA adapters to the model...")
    model = apply_lora_to_model(
        model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
    )

    # Load and preprocess data
    print(f"Loading data for pair: {pair_name}")
    dataset = load_data_for_pair(pair_name)
    preprocess_fn = create_preprocess_fn(tokenizer, src_lang_token, tgt_lang_token, max_length)
    tokenized = dataset.map(preprocess_fn, batched=True)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        predict_with_generate=True,
        logging_dir=str(LOGS_DIR / output_dir.name),
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        fp16=fp16 and torch.cuda.is_available(),
        report_to="none",
    )

    compute_metrics_fn = create_compute_metrics(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
    )

    print("Starting training (LoRA adapters will be updated)...")
    train_result = trainer.train()

    # Save the adapter (PEFT) and tokenizer
    print("Saving adapter and tokenizer...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # If model is a peft model, use save_pretrained to save adapters
    try:
        model.save_pretrained(str(output_dir))
    except Exception:
        # Fallback to trainer.save_model
        trainer.save_model(str(output_dir))

    tokenizer.save_pretrained(str(output_dir))

    eval_results = trainer.evaluate()

    results = {
        "pair": pair_name,
        "model_source": model_name_or_path,
        "train_loss": float(train_result.training_loss) if hasattr(train_result, "training_loss") else None,
        "eval": eval_results,
    }

    with open(output_dir / "lora_training_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Training complete. Results saved to {output_dir}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train mBART with LoRA adapters on a single pair")
    parser.add_argument("--pair", type=str, required=True, help="Language pair folder name (e.g., en-tl)")
    parser.add_argument("--model", type=str, default="facebook/mbart-large-50", help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save adapter + tokenizer")
    parser.add_argument("--src_lang", type=str, default="en_XX", help="Tokenizer src language tag (e.g., en_XX)")
    parser.add_argument("--tgt_lang", type=str, default="tl_XX", help="Tokenizer tgt language tag (e.g., tl_XX)")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if LORA_IMPORT_ERROR is not None:
        print("ERROR: peft import failed:", LORA_IMPORT_ERROR)
        print("Install `peft` via: pip install peft")
        sys.exit(1)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = train_pair_with_lora(
        pair_name=args.pair,
        model_name_or_path=args.model,
        output_dir=outdir,
        src_lang_token=args.src_lang,
        tgt_lang_token=args.tgt_lang,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        fp16=args.fp16,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
