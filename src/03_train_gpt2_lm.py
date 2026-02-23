"""Fine-tune GPT-2 as a comment language model (M2 generator).

Run:
  python src/03_train_gpt2_lm.py --config configs/mozilla.yaml --workdir outputs/mozilla

Notes:
- Uses TRAIN split only.
- Saves checkpoint to outputs/mozilla/gpt2_lm
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from utils import ensure_dir, load_config, read_json, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model", default=None, help="Base model name (default from config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gen_cfg = cfg.get("generator", {})
    seed = int(gen_cfg.get("seed", cfg["dataset"]["split"]["seed"]))
    seed_everything(seed)

    workdir = args.workdir
    clean_path = os.path.join(workdir, "clean.json")
    split_dir = os.path.join(workdir, "splits")

    df = pd.DataFrame(read_json(clean_path))
    cols = cfg["dataset"]["columns"]

    train_ids = np.array(read_json(os.path.join(split_dir, "train_ids.json")), dtype=np.int64)
    train_df = df.iloc[train_ids].reset_index(drop=True)

    text_col = cols["comments"]
    train_texts = train_df[text_col].fillna("").astype(str).tolist()

    base_model = args.model or gen_cfg.get("model_name", "gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds = Dataset.from_dict({"text": train_texts})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    tok = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    out_dir = os.path.join(workdir, "gpt2_lm")
    ensure_dir(out_dir)

    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=200,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved GPT-2 LM to: {out_dir}")


if __name__ == "__main__":
    main()
