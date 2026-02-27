#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_train_eval.py  (v7: adds M1)

Modes:
- m1: baseline (clean train split only, NO synthetic)
- m2: PRO_BUG (augmented/m2_rXX.jsonl)
- m3: RAG synthetic (augmented/m3_rXX_kK.jsonl)

Splits:
- workdir/splits/train_ids.json + test_ids.json are ROW INDICES into workdir/clean.json

Saves metrics in SAME FORMAT as your previous:
M1:
  metrics/m1_LinearSVC.json, ... , metrics/m1_ALL.json
M2:
  metrics/m2_r50_LinearSVC.json, ... , metrics/m2_r50_ALL.json
M3:
  metrics/m3_r05_k1_LinearSVC.json, ... , metrics/m3_r05_k1_ALL.json

Also writes per-model .txt reports.

If TensorFlow is not available, CNN/BiLSTM are skipped with a warning.
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

try:
    import yaml
except Exception:
    yaml = None

# Optional deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding, Conv1D, GlobalMaxPooling1D, Dense,
        Dropout, LSTM, Bidirectional
    )
except Exception:
    tf = None


# ---------------------------
# Logging
# ---------------------------
def setup_logging(workdir: Path) -> Path:
    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train_eval_debug.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info("Logging initialized")
    logging.info(f"Log file: {log_file}")
    return log_file


# ---------------------------
# Config + IO
# ---------------------------
def load_yaml_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not available. Install with: pip install pyyaml")
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML did not parse into a dict.")
    return cfg


def _parse_jsonl_lines(text: str, path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            logging.error(f"Bad JSONL at line {i} in {path}: {e}")
            raise
    return rows


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    logging.debug(f"Attempting to load file: {path}")
    if not path.exists():
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        logging.warning(f"File empty: {path}")
        return []

    # Looks like JSON (array/object) — try json.loads first
    if text[0] in ["[", "{"]:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                data = [data]
            logging.debug(f"Loaded JSON value: {len(data)} rows")
            return data
        except json.JSONDecodeError as e:
            # Common case: JSONL file starting with '{' => "Extra data"
            logging.debug(f"json.loads failed for {path} ({e}); trying JSONL parse.")
            rows = _parse_jsonl_lines(text, path)
            logging.debug(f"Loaded JSONL rows: {len(rows)}")
            return rows

    # Otherwise treat as JSONL
    rows = _parse_jsonl_lines(text, path)
    logging.debug(f"Loaded JSONL rows: {len(rows)}")
    return rows


# ---------------------------
# Paths
# ---------------------------
def find_aug_file(workdir: Path, mode: str, ratio: str, k: int) -> Path:
    aug_dir = workdir / "augmented"
    if mode == "m2":
        return aug_dir / f"m2_{ratio}.jsonl"
    if mode == "m3":
        return aug_dir / f"m3_{ratio}_k{k}.jsonl"
    raise ValueError(f"Mode {mode} has no augmented file (m1 baseline).")


# ---------------------------
# Dataset shaping
# ---------------------------
def make_text_series(df: pd.DataFrame, summary_col: str, comments_col: str) -> pd.Series:
    s = df.get(summary_col, pd.Series([""] * len(df))).fillna("").astype(str)
    c = df.get(comments_col, pd.Series([""] * len(df))).fillna("").astype(str)
    return (s + "\n" + c).astype(str)


def normalize_text_label(df: pd.DataFrame, summary_col: str, comments_col: str, label_col: str) -> pd.DataFrame:
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found. Available: {sorted(df.columns)}")
    out = pd.DataFrame()
    out["text"] = make_text_series(df, summary_col, comments_col)
    out["label"] = df[label_col].fillna("UNKNOWN").astype(str)
    return out.dropna()


def load_base_splits(workdir: Path, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clean_path = workdir / "clean.json"
    split_dir = workdir / "splits"
    train_ids_path = split_dir / "train_ids.json"
    test_ids_path = split_dir / "test_ids.json"

    logging.info(f"Loading clean dataset: {clean_path}")
    clean_rows = load_json_or_jsonl(clean_path)
    df = pd.DataFrame(clean_rows)

    cols = cfg["dataset"]["columns"]
    summary_col = cols["summary"]
    comments_col = cols["comments"]
    label_col = cols["label"]

    logging.info(f"Loading train ids: {train_ids_path}")
    train_ids = pd.Series(load_json_or_jsonl(train_ids_path)).astype(int).to_numpy()

    logging.info(f"Loading test ids: {test_ids_path}")
    test_ids = pd.Series(load_json_or_jsonl(test_ids_path)).astype(int).to_numpy()

    logging.info(f"Selecting splits by row index (iloc): train={len(train_ids)} test={len(test_ids)}")
    train_df_raw = df.iloc[train_ids].reset_index(drop=True)
    test_df_raw = df.iloc[test_ids].reset_index(drop=True)

    train_df = normalize_text_label(train_df_raw, summary_col, comments_col, label_col)
    test_df = normalize_text_label(test_df_raw, summary_col, comments_col, label_col)

    logging.info(f"Base train rows: {len(train_df)}")
    logging.info(f"Base test rows: {len(test_df)}")
    return train_df, test_df


def load_aug_df(workdir: Path, cfg: Dict[str, Any], mode: str, ratio: str, k: int) -> pd.DataFrame:
    aug_path = find_aug_file(workdir, mode, ratio, k)
    logging.info(f"Loading augmented: {aug_path}")
    if not aug_path.exists():
        raise FileNotFoundError(aug_path)

    rows = load_json_or_jsonl(aug_path)

    cols = cfg["dataset"]["columns"]
    summary_col = cols["summary"]
    comments_col = cols["comments"]
    label_col = cols["label"]

    return normalize_text_label(pd.DataFrame(rows), summary_col, comments_col, label_col)


# ---------------------------
# Save metrics (same naming style)
# ---------------------------
def metrics_base_name(mode: str, ratio: str, k: int, model_name: str) -> str:
    if mode == "m1":
        return f"m1_{model_name}"
    if mode == "m3":
        return f"{mode}_{ratio}_k{k}_{model_name}"
    return f"{mode}_{ratio}_{model_name}"


def save_metrics(workdir: Path, base: str, payload: Dict[str, Any], report_text: str) -> None:
    metrics_dir = workdir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    out_json = metrics_dir / f"{base}.json"
    out_txt = metrics_dir / f"{base}.txt"

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_txt.write_text(report_text, encoding="utf-8")

    logging.info(f"Saved metrics JSON: {out_json}")
    logging.info(f"Saved report TXT: {out_txt}")


def summarize_report(report_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "accuracy": report_dict.get("accuracy"),
        "macro_avg": report_dict.get("macro avg", {}),
        "weighted_avg": report_dict.get("weighted avg", {}),
    }


# ---------------------------
# Classical models (TF-IDF)
# ---------------------------
def run_tfidf_model(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=30000)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    if model_name == "LinearSVC":
        model = LinearSVC()
    elif model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=2000)
    else:
        raise ValueError(f"Unknown TF-IDF model: {model_name}")

    logging.info(f"Training {model_name}...")
    model.fit(X_train, train_df["label"])

    logging.info(f"Predicting {model_name}...")
    preds = model.predict(X_test)

    acc = accuracy_score(test_df["label"], preds)
    rep_text = classification_report(test_df["label"], preds, digits=4, zero_division=0)
    rep_dict = classification_report(test_df["label"], preds, output_dict=True, zero_division=0)
    return acc, rep_text, rep_dict


# ---------------------------
# Deep models (Keras) — optional
# ---------------------------
def encode_labels(train_labels: pd.Series, test_labels: pd.Series):
    uniq = sorted(set(train_labels.astype(str).tolist()))
    label2id = {l: i for i, l in enumerate(uniq)}
    id2label = {i: l for l, i in label2id.items()}
    y_train = np.array([label2id.get(l, -1) for l in train_labels.astype(str).tolist()], dtype=np.int32)
    y_test = np.array([label2id.get(l, -1) for l in test_labels.astype(str).tolist()], dtype=np.int32)
    return y_train, y_test, label2id, id2label


def run_keras_model(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
                    max_len: int, vocab_size: int, embed_dim: int, epochs: int, batch_size: int):
    if tf is None:
        raise RuntimeError("TensorFlow not available")

    tok = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tok.fit_on_texts(train_df["text"].tolist())

    X_train = pad_sequences(
        tok.texts_to_sequences(train_df["text"].tolist()),
        maxlen=max_len, padding="post", truncating="post"
    )
    X_test = pad_sequences(
        tok.texts_to_sequences(test_df["text"].tolist()),
        maxlen=max_len, padding="post", truncating="post"
    )

    y_train, y_test, label2id, id2label = encode_labels(train_df["label"], test_df["label"])
    n_classes = len(label2id)

    # Ignore unseen labels in test (rare)
    valid = (y_test >= 0)
    X_test_v = X_test[valid]
    y_test_v = y_test[valid]

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len))

    if model_name == "CNN":
        model.add(Conv1D(128, 5, activation="relu"))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.2))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(n_classes, activation="softmax"))
    elif model_name == "BiLSTM":
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(n_classes, activation="softmax"))
    else:
        raise ValueError(model_name)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    logging.info(f"Training {model_name} (Keras)...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    logging.info(f"Predicting {model_name} (Keras)...")
    probs = model.predict(X_test_v, verbose=0)
    pred_ids = probs.argmax(axis=1)

    y_true = [id2label[i] for i in y_test_v.tolist()]
    y_pred = [id2label[i] for i in pred_ids.tolist()]

    acc = accuracy_score(y_true, y_pred)
    rep_text = classification_report(y_true, y_pred, digits=4, zero_division=0)
    rep_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return acc, rep_text, rep_dict


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--mode", required=True, choices=["m1", "m2", "m3"])
    ap.add_argument("--ratio", default="base", help="Required for m2/m3; ignored for m1 (default: base).")
    ap.add_argument("--all_k", nargs="*", type=int, default=[5], help="Only used for m3.")
    ap.add_argument("--aug_only", type=int, default=0)

    ap.add_argument("--models", nargs="*", default=["all"],
                    help="Models to run: all OR any of LinearSVC LogisticRegression CNN BiLSTM")

    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--vocab_size", type=int, default=30000)
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    workdir = Path(args.workdir)
    setup_logging(workdir)

    allowed = ["LinearSVC", "LogisticRegression", "CNN", "BiLSTM"]
    if len(args.models) == 1 and args.models[0].lower() == "all":
        model_list = allowed
    else:
        model_list = args.models
        for m in model_list:
            if m not in allowed:
                raise ValueError(f"Unknown model '{m}'. Allowed: {allowed} or 'all'")

    try:
        cfg = load_yaml_config(Path(args.config))
        base_train_df, test_df = load_base_splits(workdir, cfg)

        # M1: single run, no K loop, no augmentation
        if args.mode == "m1":
            ks = [0]
        elif args.mode == "m2":
            ks = [0]  # no k for m2
        else:
            ks = args.all_k  # m3

        for k in ks:
            logging.info("##############################")
            logging.info(f"RUN: mode={args.mode} ratio={args.ratio} k={k}")
            logging.info("##############################")

            if args.mode == "m1":
                train_df = base_train_df

            else:
                aug_df = load_aug_df(workdir, cfg, args.mode, args.ratio, k)
                train_df = aug_df if args.aug_only else pd.concat([base_train_df, aug_df], ignore_index=True)

            all_summary: Dict[str, Any] = {
                "mode": args.mode,
                "ratio": args.ratio if args.mode != "m1" else None,
                "k": k if args.mode == "m3" else None,
                "models": {}
            }

            for model_name in model_list:
                try:
                    if model_name in ["LinearSVC", "LogisticRegression"]:
                        acc, rep_text, rep_dict = run_tfidf_model(model_name, train_df, test_df)
                    else:
                        if tf is None:
                            logging.warning(f"Skipping {model_name}: TensorFlow not available.")
                            continue
                        acc, rep_text, rep_dict = run_keras_model(
                            model_name, train_df, test_df,
                            max_len=args.max_len,
                            vocab_size=args.vocab_size,
                            embed_dim=args.embed_dim,
                            epochs=args.epochs,
                            batch_size=args.batch_size
                        )

                    base = metrics_base_name(args.mode, args.ratio, k, model_name)
                    payload = {
                        "mode": args.mode,
                        "ratio": args.ratio if args.mode != "m1" else None,
                        "k": k if args.mode == "m3" else None,
                        "model": model_name,
                        "accuracy": acc,
                        "macro_avg": rep_dict.get("macro avg", {}),
                        "weighted_avg": rep_dict.get("weighted avg", {}),
                        "per_class": {kk: vv for kk, vv in rep_dict.items()
                                      if kk not in ["accuracy", "macro avg", "weighted avg"]},
                    }
                    save_metrics(workdir, base, payload, rep_text)
                    all_summary["models"][model_name] = summarize_report(rep_dict)

                except Exception as e:
                    logging.error(f"Model {model_name} failed: {e}")
                    logging.error(traceback.format_exc())

            all_base = metrics_base_name(args.mode, args.ratio, k, "ALL")
            save_metrics(workdir, all_base, all_summary, json.dumps(all_summary, indent=2))

    except Exception as e:
        logging.error("Fatal error occurred")
        logging.error(str(e))
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
