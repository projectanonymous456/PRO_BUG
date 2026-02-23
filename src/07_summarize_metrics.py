#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
07_summarize_metrics.py (v3)

Goal:
- Collect Precision / Recall / (Micro, Macro) / Accuracy for ALL 3 modes (m1, m2, m3)
- For ALL models: LinearSVC, LogisticRegression, BiLSTM, CNN
- Works with BOTH naming styles you have used across versions:

UNDERSCORE style (from 06_train_eval.py v7):
  m1_ALL.json
  m1_LinearSVC.json
  m2_r50_ALL.json
  m2_r50_LinearSVC.json
  m3_r05_k5_ALL.json
  m3_r05_k5_LinearSVC.json

DOUBLE-UNDERSCORE style (older summarize script):
  m3__r50__k5__LinearSVC.json
  m3__r50__k5__ALL.json

Reads BOTH:
- aggregated "*_ALL.json" (contains metrics for multiple models)
- per-model JSONs ("*_LinearSVC.json", etc.)

Outputs:
- outdir/summary.xlsx
- outdir/summary.csv

Notes on "micro" for multiclass:
scikit-learn's classification_report output_dict for multiclass usually does NOT include "micro avg".
For single-label multiclass, micro-precision = micro-recall = micro-F1 = accuracy.
So this script sets micro_* = accuracy when micro avg is missing (common case).
"""

import os
import json
import glob
import re
import argparse
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# ----------------------------
# Patterns (support _ and __)
# ----------------------------

# Aggregated ALL (underscore style):
#   m1_ALL.json
#   m2_r50_ALL.json
#   m3_r05_k5_ALL.json
PAT_ALL_US = re.compile(
    r"^(m[123])(?:_(r\d+))?(?:_k(\d+))?_ALL\.json$",
    re.IGNORECASE
)

# Per-model (underscore style):
#   m1_LinearSVC.json
#   m2_r50_LinearSVC.json
#   m3_r05_k5_LinearSVC.json
PAT_MODEL_US = re.compile(
    r"^(m[123])(?:_(r\d+))?(?:_k(\d+))?_(LinearSVC|LogisticRegression|BiLSTM|CNN)\.json$",
    re.IGNORECASE
)

# Aggregated ALL (double-underscore style):
#   m3__r50__k5__ALL.json
#   m2__r10__ALL.json  (if ever)
PAT_ALL_DU = re.compile(
    r"^(m[123])(?:__?(r\d+))?(?:__?k(\d+))?__ALL\.json$",
    re.IGNORECASE
)

# Per-model (double-underscore style):
#   m3__r50__k5__LinearSVC.json
PAT_MODEL_DU = re.compile(
    r"^(m[123])__(r\d+)(?:__k(\d+))?__(LinearSVC|LogisticRegression|BiLSTM|CNN)\.json$",
    re.IGNORECASE
)

KNOWN_MODELS = ["LinearSVC", "LogisticRegression", "BiLSTM", "CNN"]


def _safe_get(dct: Any, *keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ratio_to_pct(ratio_str: Optional[str]) -> Optional[int]:
    # "r05" -> 5, "r10" -> 10, ...
    if not ratio_str:
        return None
    m = re.match(r"r(\d+)", str(ratio_str).strip(), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _normalize_model_name(name: str) -> str:
    name = (name or "").strip()
    for km in KNOWN_MODELS:
        if km.lower() == name.lower():
            return km
    return name


def _key(dataset_label: str, mode: str, ratio: Optional[str], k: Optional[int]) -> Tuple[str, str, Optional[str], Optional[int]]:
    return (dataset_label, (mode or "").upper(), (ratio or "").lower() if ratio else None, int(k) if k is not None else None)


def _fill_micro_from_accuracy(row: Dict[str, Any], model: str):
    """
    For single-label multiclass:
    micro-precision = micro-recall = micro-f1 = accuracy
    """
    acc = row.get(f"{model}_accuracy")
    if acc is None:
        return
    for metric in ["micro_precision", "micro_recall", "micro_f1"]:
        col = f"{model}_{metric}"
        if row.get(col) is None:
            row[col] = acc


def _extract_from_reportlike(row: Dict[str, Any], model: str, obj: Dict[str, Any]):
    """
    obj is expected to be a per-model JSON produced by 06_train_eval.py v7:
      {
        "accuracy": ...,
        "macro_avg": {"precision":..., "recall":..., "f1-score":..., "support":...},
        "weighted_avg": {...},
        ...
      }
    or an ALL.json model entry with:
      {"accuracy":..., "macro_avg":{...}, "weighted_avg":{...}}
    """
    model = _normalize_model_name(model)

    acc = obj.get("accuracy")
    if acc is not None:
        row[f"{model}_accuracy"] = acc

    macro = obj.get("macro_avg") or obj.get("macro avg") or {}
    if isinstance(macro, dict):
        row[f"{model}_macro_precision"] = macro.get("precision", row.get(f"{model}_macro_precision"))
        row[f"{model}_macro_recall"] = macro.get("recall", row.get(f"{model}_macro_recall"))
        row[f"{model}_macro_f1"] = macro.get("f1-score", macro.get("f1", row.get(f"{model}_macro_f1")))

    micro = obj.get("micro_avg") or obj.get("micro avg") or {}
    if isinstance(micro, dict) and micro:
        row[f"{model}_micro_precision"] = micro.get("precision", row.get(f"{model}_micro_precision"))
        row[f"{model}_micro_recall"] = micro.get("recall", row.get(f"{model}_micro_recall"))
        row[f"{model}_micro_f1"] = micro.get("f1-score", micro.get("f1", row.get(f"{model}_micro_f1")))

    # If micro not present (typical multiclass), fill from accuracy
    _fill_micro_from_accuracy(row, model)


def _extract_from_top_level_f1(row: Dict[str, Any], model: str, obj: Dict[str, Any]):
    """
    Older per-model JSONs sometimes store:
      {"macro_f1": ..., "micro_f1": ...}
    plus maybe "accuracy". Keep compatibility.
    """
    model = _normalize_model_name(model)
    if "accuracy" in obj and obj["accuracy"] is not None:
        row[f"{model}_accuracy"] = obj["accuracy"]

    if "macro_f1" in obj and obj["macro_f1"] is not None:
        row[f"{model}_macro_f1"] = obj["macro_f1"]
    if "micro_f1" in obj and obj["micro_f1"] is not None:
        row[f"{model}_micro_f1"] = obj["micro_f1"]

    # If only micro_f1 exists and accuracy missing, keep micro_f1; otherwise fill from accuracy
    _fill_micro_from_accuracy(row, model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdirs", nargs="+", required=True, help="List of workdirs")
    ap.add_argument("--labels", nargs="+", required=True, help="Dataset labels (same length as workdirs)")
    ap.add_argument("--outdir", default="summary", help="Output folder for summary.xlsx/csv")
    args = ap.parse_args()

    if len(args.workdirs) != len(args.labels):
        raise SystemExit("ERROR: --workdirs and --labels must have the same length")

    rows_by_key: Dict[Tuple[str, str, Optional[str], Optional[int]], Dict[str, Any]] = {}

    for wd, dataset_label in zip(args.workdirs, args.labels):
        metrics_dir = os.path.join(wd, "metrics")

        search_dirs = []
        if os.path.isdir(metrics_dir):
            search_dirs.append(metrics_dir)
        search_dirs.append(wd)

        json_files = []
        for d in search_dirs:
            json_files.extend(glob.glob(os.path.join(d, "*.json")))

        if not json_files:
            print(f"[WARN] No JSON files found under: {metrics_dir} or {wd}")
            continue

        for p in sorted(set(json_files)):
            base = os.path.basename(p)

            m_all = PAT_ALL_US.match(base) or PAT_ALL_DU.match(base)
            m_mod = PAT_MODEL_US.match(base) or PAT_MODEL_DU.match(base)

            if not (m_all or m_mod):
                continue

            try:
                obj = _read_json(p)
            except Exception as e:
                print(f"[WARN] Could not read {p}: {e}")
                continue

            if m_all:
                mode = (m_all.group(1) or "").upper()
                ratio = (m_all.group(2) or obj.get("ratio") or None)
                ratio = ratio.lower() if isinstance(ratio, str) and ratio else None
                k = m_all.group(3)
                k = int(k) if k is not None else obj.get("k", None)

                key = _key(dataset_label, mode, ratio, k)
                row = rows_by_key.get(key, {
                    "dataset": dataset_label,
                    "mode": mode,
                    "ratio": ratio,
                    "syn_ratio_pct": _ratio_to_pct(ratio),
                    "k": k,
                    "metrics_all_file": None,
                    "metrics_files": [],
                })

                row["metrics_all_file"] = p
                row["metrics_files"].append(p)

                # aggregated file structure in 06_train_eval.py v7:
                # {"mode":..., "ratio":..., "k":..., "models": { "LinearSVC": {"accuracy":..., "macro_avg":..., ...}, ...}}
                models = obj.get("models", {}) if isinstance(obj.get("models", {}), dict) else {}
                for model_name, model_payload in models.items():
                    model_name = _normalize_model_name(model_name)
                    if model_name not in KNOWN_MODELS:
                        continue
                    if isinstance(model_payload, dict):
                        _extract_from_reportlike(row, model_name, model_payload)

                rows_by_key[key] = row
                continue

            if m_mod:
                mode = (m_mod.group(1) or "").upper()
                ratio = m_mod.group(2)
                ratio = ratio.lower() if isinstance(ratio, str) and ratio else None
                k = m_mod.group(3)
                k = int(k) if k is not None else None
                model_name = _normalize_model_name(m_mod.group(4) or "")

                key = _key(dataset_label, mode, ratio, k)
                row = rows_by_key.get(key, {
                    "dataset": dataset_label,
                    "mode": mode,
                    "ratio": ratio,
                    "syn_ratio_pct": _ratio_to_pct(ratio),
                    "k": k,
                    "metrics_all_file": None,
                    "metrics_files": [],
                })

                row["metrics_files"].append(p)

                # Prefer reportlike extraction (newer), fallback to top-level f1 (older)
                if isinstance(obj, dict) and ("macro_avg" in obj or "macro avg" in obj or "accuracy" in obj):
                    _extract_from_reportlike(row, model_name, obj)
                else:
                    _extract_from_top_level_f1(row, model_name, obj)

                rows_by_key[key] = row
                continue

    df = pd.DataFrame(list(rows_by_key.values()))
    os.makedirs(args.outdir, exist_ok=True)

    out_xlsx = os.path.join(args.outdir, "summary.xlsx")
    out_csv = os.path.join(args.outdir, "summary.csv")

    # Column ordering
    pref = ["dataset", "mode", "syn_ratio_pct", "ratio", "k", "metrics_all_file", "metrics_files"]
    metric_cols = []
    for mn in KNOWN_MODELS:
        metric_cols += [
            f"{mn}_accuracy",
            f"{mn}_macro_precision", f"{mn}_macro_recall", f"{mn}_macro_f1",
            f"{mn}_micro_precision", f"{mn}_micro_recall", f"{mn}_micro_f1",
        ]

    cols = [c for c in pref if c in df.columns] + [c for c in metric_cols if c in df.columns]
    cols += [c for c in df.columns if c not in set(cols)]
    if len(df.columns):
        df = df[cols]

    df.to_excel(out_xlsx, index=False)
    df.to_csv(out_csv, index=False)

    # Console preview
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(df.to_string(index=False))

    print(f"\n[OK] Wrote: {out_xlsx}")
    print(f"[OK] Wrote: {out_csv}")


if __name__ == "__main__":
    main()
