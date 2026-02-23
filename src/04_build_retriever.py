"""Build train-only embeddings + FAISS index (prevents leakage).

Run:
  python src/04_build_retriever.py --config configs/mozilla.yaml --workdir outputs/mozilla

Outputs:
  outputs/mozilla/retriever/index.faiss
  outputs/mozilla/retriever/train_ids.npy
  outputs/mozilla/retriever/train_embeddings.npy
"""

import argparse
import os

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from utils import ensure_dir, load_config, read_json, seed_everything


def _make_corpus_text(df: pd.DataFrame, summary_col: str, comments_col: str) -> list[str]:
    s = df[summary_col].fillna("").astype(str)
    c = df[comments_col].fillna("").astype(str)
    return (s + "\n" + c).tolist()


def build_hnsw_index(xb: np.ndarray) -> faiss.Index:
    dim = xb.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 100
    index.add(xb)
    index.hnsw.efSearch = 64
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("generator", {}).get("seed", cfg["dataset"]["split"]["seed"]))
    seed_everything(seed)

    workdir = args.workdir
    clean_path = os.path.join(workdir, "clean.json")
    split_dir = os.path.join(workdir, "splits")

    data = read_json(clean_path)
    df = pd.DataFrame(data)

    cols = cfg["dataset"]["columns"]
    train_ids = np.array(read_json(os.path.join(split_dir, "train_ids.json")), dtype=np.int64)
    train_df = df.iloc[train_ids].reset_index(drop=True)

    retr_cfg = cfg["retriever"]
    embed_model = retr_cfg["embed_model"]

    out_dir = os.path.join(workdir, "retriever")
    ensure_dir(out_dir)

    corpus_texts = _make_corpus_text(train_df, cols["summary"], cols["comments"])

    embedder = SentenceTransformer(embed_model)
    xb = embedder.encode(corpus_texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    # Train-only index
    index = build_hnsw_index(xb)

    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    np.save(os.path.join(out_dir, "train_embeddings.npy"), xb)
    np.save(os.path.join(out_dir, "train_row_ids.npy"), train_ids)

    print(f"Saved FAISS index to: {os.path.join(out_dir, 'index.faiss')}")
    print(f"Train rows indexed: {len(train_ids)}")


if __name__ == "__main__":
    main()
