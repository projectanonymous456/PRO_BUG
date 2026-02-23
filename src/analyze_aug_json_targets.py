import os
import json
import argparse
from collections import Counter
from typing import Optional, List

import numpy as np
import pandas as pd


BUG_ID_CANDIDATES = ["bug_id", "BugId", "bugId", "id", "ID"]
LABEL_CANDIDATES = ["label", "assignee", "developer", "owner", "assigned_to", "AssignedTo"]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_target(counts: Counter, policy: str, cap: int) -> int:
    arr = np.array(list(counts.values()), dtype=np.int64)
    if len(arr) == 0:
        return 0
    if policy == "median":
        target = int(np.median(arr))
    elif policy == "mean":
        target = int(np.mean(arr))
    else:
        raise ValueError(f"Unknown policy: {policy}")
    target = max(1, min(target, int(cap)))
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json_path",
        required=True,
        help="Path to ANY train/augmented json list file (e.g., .../augmented/m3_k5_train.json)",
    )
    ap.add_argument("--minority_threshold", type=int, default=10)
    ap.add_argument("--target_policy", choices=["median", "mean"], default="median")
    ap.add_argument("--target_cap", type=int, default=200)
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--out_csv", default=None, help="Optional output CSV path")
    args = ap.parse_args()

    path = args.json_path
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data = load_json(path)
    if not isinstance(data, list) or len(data) == 0:
        print("❌ File is not a non-empty list-of-records JSON:", path)
        return

    df = pd.DataFrame(data)

    label_col = detect_col(df, LABEL_CANDIDATES)
    bug_col = detect_col(df, BUG_ID_CANDIDATES)

    if label_col is None:
        print("❌ Could not detect label column.")
        print("Columns:", list(df.columns))
        return

    # synthetic detector
    if "is_synthetic" in df.columns:
        is_syn = df["is_synthetic"].fillna(False).astype(bool)
    elif bug_col is not None:
        is_syn = df[bug_col].astype(str).str.startswith("AI_")
    else:
        # if no field exists, we cannot reliably separate
        is_syn = pd.Series([False] * len(df))

    df["__is_syn__"] = is_syn

    # counts
    counts_all = Counter(df[label_col].astype(str).tolist())
    counts_orig = Counter(df.loc[~df["__is_syn__"], label_col].astype(str).tolist())
    counts_syn = Counter(df.loc[df["__is_syn__"], label_col].astype(str).tolist())

    target = compute_target(counts_all, args.target_policy, args.target_cap)

    minority = {lab: n for lab, n in counts_all.items() if n < args.minority_threshold}
    need_more = {lab: max(0, target - n) for lab, n in counts_all.items() if n < args.minority_threshold}

    rows = []
    for lab in minority.keys():
        rows.append(
            {
                "label": lab,
                "count_total": counts_all.get(lab, 0),
                "count_original": counts_orig.get(lab, 0),
                "count_synthetic": counts_syn.get(lab, 0),
                "target_per_label": target,
                "intended_more_to_generate": need_more.get(lab, 0),
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["count_total", "intended_more_to_generate"], ascending=[True, False]
    )

    print("=" * 100)
    print("FILE:", path)
    print("rows total    :", len(df))
    print("rows original :", int((~df["__is_syn__"]).sum()) if "__is_syn__" in df else "unknown")
    print("rows synthetic:", int(df["__is_syn__"].sum()) if "__is_syn__" in df else "unknown")
    print("label column  :", label_col)
    print("bug_id column :", bug_col)
    print("-" * 100)
    print("minority_threshold:", args.minority_threshold)
    print("target_policy     :", args.target_policy)
    print("target_cap        :", args.target_cap)
    print("computed target   :", target)
    print("num minority labels:", len(minority))
    print("planned additional synthetics (from this file’s counts):", int(sum(need_more.values())))
    print("=" * 100)

    if out.empty:
        print("No minority classes under threshold.")
    else:
        print(out.head(args.topn).to_string(index=False))

    # output csv
    if args.out_csv:
        out_csv = args.out_csv
    else:
        base = os.path.splitext(os.path.basename(path))[0]
        out_csv = os.path.join(os.path.dirname(path), f"{base}__minority_targets.csv")

    out.to_csv(out_csv, index=False)
    print("\n[OK] Saved:", out_csv)


if __name__ == "__main__":
    main()