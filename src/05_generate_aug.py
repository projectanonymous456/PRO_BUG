#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
M1 - Base
M2 - PRO_BUG
M3 - RAG

Notes
- Assumes your workdir has:
    <workdir>/clean.json
    <workdir>/splits/train_ids.json
- Uses YAML config for dataset columns and ratios:
    cfg["dataset"]["columns"], cfg["augment"]["syn_ratios"]
"""

import argparse
import os
import json
import logging
import traceback
import time
import hashlib
import shutil
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional memory info
try:
    import psutil
except Exception:
    psutil = None

# Retriever deps
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except Exception as e:
    faiss = None
    SentenceTransformer = None
    _IMPORT_ERR = repr(e)

MODEL_NAME_DEFAULT = "mistralai/Mistral-7B-Instruct-v0.2"
M3_SAFE_EMB_FALLBACK = "BAAI/bge-base-en-v1.5"


# ============================================================
# Logging
# ============================================================

def setup_logger(workdir: str, mode: str) -> logging.Logger:
    log_dir = os.path.join(workdir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"generate_{mode}_{ts}.log")
    err_path = os.path.join(log_dir, f"generate_{mode}_{ts}.error.log")

    logger = logging.getLogger("generate_aug")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    eh = logging.FileHandler(err_path)
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.addHandler(eh)

    logger.info(f"Logging to: {log_path}")
    logger.info(f"Error log: {err_path}")
    return logger


def log_memory(logger: logging.Logger, tag: str = "") -> None:
    if psutil:
        mem = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        logger.info(f"Memory {tag}: {mem:.2f} GB")


def log_gpu(logger: logging.Logger, tag: str = "") -> None:
    if torch.cuda.is_available():
        try:
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserv = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"GPU {tag}: allocated={alloc:.2f} GB reserved={reserv:.2f} GB")
        except Exception:
            pass


# ============================================================
# JSONL Writing (local tmp first)
# ============================================================

def jsonl_write_many(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def copy_tmp_to_drive(logger: logging.Logger, tmp_path: str, drive_path: str) -> None:
    Path(drive_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tmp_path, drive_path)
    logger.info(f"[SYNC] Copied tmp -> Drive: {tmp_path} -> {drive_path}")


# ============================================================
# Minority Allocation
# ============================================================

def total_syn_needed(n_orig: int, ratio: float) -> int:
    # total_syn = ceil((r * N) / (1-r))
    return int(np.ceil((ratio * n_orig) / (1.0 - ratio)))


def pick_minority(cnt: Counter) -> List[str]:
    med = np.median(list(cnt.values()))
    return [l for l, c in cnt.items() if c < med]


def allocate_per_label(total_syn: int, labels: List[str], per_label_cap: int) -> Dict[str, int]:
    if total_syn <= 0 or not labels:
        return {}

    m = len(labels)
    base = total_syn // m
    rem = total_syn % m

    alloc: Dict[str, int] = {}
    for i, lab in enumerate(labels):
        n = base + (1 if i < rem else 0)
        n = min(n, per_label_cap)
        alloc[lab] = n
    return alloc


# ============================================================
# Mistral Instruct Prompt wrapper
# ============================================================

def wrap_mistral_inst(user_text: str) -> str:
    return f"<s>[INST]\n{user_text.strip()}\n[/INST]\n"


# ============================================================
# FAST GENERATION
# ============================================================

def _autocast_ctx():
    if torch.cuda.is_available():
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)

    class _NoOp:
        def __enter__(self): return None
        def __exit__(self, *args): return False

    return _NoOp()


def fast_generate_batch(
    model,
    tok,
    prompt: str,
    num_samples: int,
    max_prompt_tokens: int = 512,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> List[str]:
    prompt = wrap_mistral_inst(prompt)
    x = tok(prompt, return_tensors="pt", truncation=False)

    # With device_map="auto" model may span devices; use first parameter device
    device = next(model.parameters()).device

    input_ids = x["input_ids"].to(device)
    attention_mask = x["attention_mask"].to(device)

    if input_ids.shape[1] > max_prompt_tokens:
        input_ids = input_ids[:, -max_prompt_tokens:]
        attention_mask = attention_mask[:, -max_prompt_tokens:]

    input_ids = input_ids.repeat(num_samples, 1)
    attention_mask = attention_mask.repeat(num_samples, 1)

    with torch.inference_mode():
        with _autocast_ctx():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

    gen_only = outputs[:, input_ids.shape[1]:]
    texts = tok.batch_decode(gen_only, skip_special_tokens=True)
    return [t.strip() for t in texts]


# ============================================================
# Retriever (M3) with LOCAL TMP CACHE + DRIVE COPY
# ============================================================

def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def _hash_key(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"|")
    return h.hexdigest()[:12]


def _cache_key(clean_path: str, emb_model: str, index_type: str, n_texts: int) -> str:
    mtime = "0"
    try:
        mtime = str(int(os.path.getmtime(clean_path)))
    except Exception:
        pass
    return _hash_key(emb_model, index_type, mtime, str(n_texts))


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _copy_if_missing_or_newer(src: str, dst: str) -> None:
    if not os.path.exists(src):
        return
    if (not os.path.exists(dst)) or (os.path.getmtime(src) > os.path.getmtime(dst)):
        _ensure_dir(os.path.dirname(dst))
        shutil.copy2(src, dst)


def _looks_like_llm_id(model_id: str) -> bool:
    s = (model_id or "").lower()
    llm_markers = [
        "mistral", "instruct", "llama", "gemma", "qwen", "falcon", "mixtral",
        "gpt", "causal", "chat", "7b", "8b", "13b", "70b"
    ]
    hits = sum(1 for m in llm_markers if m in s)
    return hits >= 2


def build_or_load_retriever(
    corpus_texts: List[str],
    st_device: str,          # device for SentenceTransformer (default cpu)
    emb_model: str,
    index_type: str,
    clean_path: str,
    workdir_drive: str,
    logger: logging.Logger,
    cache_enabled: bool = True,
    st_batch_size: int = 512,
    tmp_root: str = "/content/rag_bug_tmp",
) -> Tuple[Any, Any, str]:
    if SentenceTransformer is None or faiss is None:
        raise RuntimeError(f"faiss/sentence-transformers missing: {_IMPORT_ERR}")

    n_texts = len(corpus_texts)
    key = _cache_key(clean_path, emb_model, index_type, n_texts)

    drive_cache_dir = os.path.join(workdir_drive, "retriever_cache")
    tmp_cache_dir = os.path.join(tmp_root, "retriever_cache")

    _ensure_dir(drive_cache_dir)
    _ensure_dir(tmp_cache_dir)

    idx_name = f"faiss_{_safe_name(index_type)}_{key}.index"
    meta_name = f"faiss_{_safe_name(index_type)}_{key}.meta.json"
    emb_name = f"emb_{_safe_name(emb_model)}_{key}.npy"

    drive_idx = os.path.join(drive_cache_dir, idx_name)
    drive_meta = os.path.join(drive_cache_dir, meta_name)
    drive_emb = os.path.join(drive_cache_dir, emb_name)

    tmp_idx = os.path.join(tmp_cache_dir, idx_name)
    tmp_meta = os.path.join(tmp_cache_dir, meta_name)
    tmp_emb = os.path.join(tmp_cache_dir, emb_name)

    # OOM FIX: SentenceTransformer is loaded on its OWN device (default: CPU)
    st = SentenceTransformer(emb_model, device=st_device)
    logger.info(f"[M3] SentenceTransformer model={emb_model} device={st_device}")
    log_gpu(logger, "after_st_load")

    # Try Drive cache first
    if cache_enabled and os.path.exists(drive_idx) and os.path.exists(drive_meta):
        try:
            meta = json.load(open(drive_meta, "r", encoding="utf-8"))
            if meta.get("n_texts") == n_texts and meta.get("emb_model") == emb_model and meta.get("index_type") == index_type:
                logger.info(f"[M3] Loading cached FAISS index from Drive: {drive_idx}")
                index = faiss.read_index(drive_idx)
                _copy_if_missing_or_newer(drive_idx, tmp_idx)
                _copy_if_missing_or_newer(drive_meta, tmp_meta)
                _copy_if_missing_or_newer(drive_emb, tmp_emb)
                return st, index, key
        except Exception as e:
            logger.info(f"[M3] Drive cache load failed, rebuilding. Reason: {repr(e)}")

    # Try tmp cache
    if cache_enabled and os.path.exists(tmp_idx) and os.path.exists(tmp_meta):
        try:
            meta = json.load(open(tmp_meta, "r", encoding="utf-8"))
            if meta.get("n_texts") == n_texts and meta.get("emb_model") == emb_model and meta.get("index_type") == index_type:
                logger.info(f"[M3] Loading cached FAISS index from local tmp: {tmp_idx}")
                index = faiss.read_index(tmp_idx)
                _copy_if_missing_or_newer(tmp_idx, drive_idx)
                _copy_if_missing_or_newer(tmp_meta, drive_meta)
                _copy_if_missing_or_newer(tmp_emb, drive_emb)
                return st, index, key
        except Exception as e:
            logger.info(f"[M3] Tmp cache load failed, rebuilding. Reason: {repr(e)}")

    # Build (on tmp)
    logger.info("[M3] Building embeddings + FAISS index (first-time build; slow) ...")
    t0 = time.time()

    embs = None
    if cache_enabled and os.path.exists(tmp_emb):
        try:
            embs = np.load(tmp_emb)
            if not isinstance(embs, np.ndarray) or embs.shape[0] != n_texts:
                embs = None
            else:
                logger.info(f"[M3] Loaded cached embeddings from tmp: {tmp_emb} shape={embs.shape}")
        except Exception:
            embs = None

    if embs is None:
        with torch.inference_mode():
            embs = st.encode(
                corpus_texts,
                batch_size=max(8, int(st_batch_size)),
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
        log_gpu(logger, "after_encode")
        if cache_enabled:
            try:
                np.save(tmp_emb, embs)
                logger.info(f"[M3] Saved embeddings cache to tmp: {tmp_emb}")
            except Exception as e:
                logger.info(f"[M3] Embedding tmp save failed (non-fatal): {repr(e)}")

    dim = embs.shape[1]
    it = (index_type or "hnsw").lower()
    if it == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
    elif it == "flatip":
        index = faiss.IndexFlatIP(dim)
    elif it == "flatl2":
        index = faiss.IndexFlatL2(dim)
    else:
        logger.info(f"[M3] Unknown index_type={index_type}, falling back to hnsw")
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64

    index.add(embs)
    logger.info(f"[M3] Index built in {time.time()-t0:.1f}s | dim={dim} | n={n_texts}")

    if cache_enabled:
        try:
            faiss.write_index(index, tmp_idx)
            meta = {
                "n_texts": n_texts,
                "dim": int(dim),
                "emb_model": emb_model,
                "index_type": index_type,
                "built_at": datetime.now().isoformat(),
            }
            json.dump(meta, open(tmp_meta, "w", encoding="utf-8"), indent=2)
            logger.info(f"[M3] Saved index+meta to tmp: {tmp_idx}")

            _copy_if_missing_or_newer(tmp_idx, drive_idx)
            _copy_if_missing_or_newer(tmp_meta, drive_meta)
            _copy_if_missing_or_newer(tmp_emb, drive_emb)
            logger.info(f"[M3] Synced retriever cache to Drive: {drive_cache_dir}")
        except Exception as e:
            logger.info(f"[M3] Cache save/sync failed (non-fatal): {repr(e)}")

    return st, index, key


def retrieve(st, index, corpus_texts: List[str], query: str, k: int, snippet_chars: int) -> List[str]:
    qv = st.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    _, ids = index.search(qv, k)
    return [corpus_texts[i][:snippet_chars] for i in ids[0]]


# ============================================================
# Leakage guards (M3 ONLY)
# ============================================================

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE)

def _sanitize_text_no_leak(text: str, label_set: Optional[set] = None) -> str:
    """Remove obvious label/PII leakage patterns from retrieved context or generated text."""
    if not text:
        return ""

    t = text

    # Remove emails
    t = _EMAIL_RE.sub("[redacted_email]", t)

    # Remove obvious assignment lines (common patterns in bug trackers)
    t = re.sub(r"(?im)^\s*(assignee|assigned\s+to)\s*[:=].*$", "", t)
    t = re.sub(r"(?im)\b(assignee|assigned\s+to)\s*[:=]\s*[^\n\r]{0,120}", "", t)

    # Remove any explicit label mentions if provided (case-insensitive whole-word)
    if label_set:
        for lab in label_set:
            lab_s = str(lab).strip()
            if not lab_s:
                continue
            t = re.sub(rf"(?i)\b{re.escape(lab_s)}\b", "[redacted_label]", t)

    # Collapse excessive whitespace
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def _detect_leak(text: str, label_set: Optional[set] = None) -> bool:
    """Heuristic leak detector: returns True if text likely contains label/PII leakage."""
    if not text:
        return False
    t = text.lower()
    if _EMAIL_RE.search(text):
        return True
    if "assigned to" in t or "assignee" in t:
        return True
    if label_set:
        for lab in label_set:
            lab_s = str(lab).strip().lower()
            if lab_s and lab_s in t:
                return True
    return False


def _sanitize_ctx_list(ctx: List[str], label_set: Optional[set] = None) -> List[str]:
    out = []
    for c in ctx:
        sc = _sanitize_text_no_leak(c, label_set=label_set)
        if sc:
            out.append(sc)
    return out


# ============================================================
# Corpus + query helpers
# ============================================================

def build_minority_corpus_df(
    train_df: pd.DataFrame,
    label_col: str,
    minority_labels: List[str],
    summary_col: Optional[str],
    comments_col: str,
    comments_cap: int = 300,
    dedup_by_summary: bool = True,
) -> pd.DataFrame:
    sub = train_df[train_df[label_col].astype(str).isin(set(minority_labels))].copy()
    if dedup_by_summary and summary_col and summary_col in sub.columns:
        sub[summary_col] = sub[summary_col].fillna("").astype(str)
        sub = sub.drop_duplicates(subset=[summary_col], keep="first").reset_index(drop=True)

    comm = sub[comments_col].fillna("").astype(str).str.slice(0, comments_cap)
    if summary_col and summary_col in sub.columns:
        summ = sub[summary_col].fillna("").astype(str)
        sub["_retr_text"] = (summ + "\n" + comm).astype(str)
    else:
        sub["_retr_text"] = comm.astype(str)

    return sub.reset_index(drop=True)


def choose_summary_for_label(
    lab: str,
    label_to_summaries: Dict[str, List[str]],
    rng: np.random.Generator
) -> str:
    """Pick a summary associated with a label (for diversity), WITHOUT ever returning the label."""
    cand = label_to_summaries.get(str(lab), [])
    if cand:
        s = cand[int(rng.integers(0, len(cand)))]
        s = (s or "").strip()
        if s:
            return s
    return ""


def parse_k_list(k_arg: Optional[Any]) -> List[int]:
    if k_arg is None:
        return [1, 3, 5]
    if isinstance(k_arg, list):
        out: List[int] = []
        for item in k_arg:
            out.extend(parse_k_list(item))
        seen = set()
        res = []
        for x in out:
            if x not in seen:
                res.append(x)
                seen.add(x)
        return res
    s = str(k_arg).strip()
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(s)]


# ============================================================
# --ratios parsing
# ============================================================

def _parse_ratio_token(tok: str) -> float:
    """
    Accept:
    - r05, r10, r25, r50
    - 0.05, 0.1, .25
    - 5, 10, 25, 50
    - 5%, 10%
    Returns float ratio in (0,1).
    """
    s = (tok or "").strip().lower()
    if not s:
        raise ValueError("empty ratio token")

    if s.startswith("r"):
        s = s[1:].strip()

    if s.endswith("%"):
        s = s[:-1].strip()

    v = float(s)
    if v >= 1.0:
        v = v / 100.0

    if not (0.0 < v < 1.0):
        raise ValueError(f"ratio out of range after parse: {tok} -> {v}")

    return float(f"{v:.2f}")


def filter_ratios_from_config(ratios_cfg: List[Any], ratios_arg: Optional[str], logger: logging.Logger) -> List[float]:
    ratios_all = [float(r) for r in ratios_cfg]
    if not ratios_arg:
        return ratios_all

    toks = []
    for chunk in ratios_arg.replace(",", " ").split():
        if chunk.strip():
            toks.append(chunk.strip())

    wanted = set(_parse_ratio_token(t) for t in toks)

    out = []
    for r in ratios_all:
        rr = float(f"{float(r):.2f}")
        if rr in wanted:
            out.append(float(r))

    logger.info(f"[RATIOS] config={ratios_all} | selected={sorted(wanted)} | will_run={out}")
    if not out:
        raise ValueError(f"--ratios requested {sorted(wanted)} but none matched config syn_ratios={ratios_all}")
    return out


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--mode", choices=["m2", "m3"], required=True)

    ap.add_argument("--k", action="append", default=None,
                    help="For m3: use --k 1 --k 3 --k 5 or --k 1,3,5. Default is 1,3,5")

    ap.add_argument("--ratios", type=str, default=None,
                    help="Run only selected ratios: e.g. r10 or r05,r10 or 0.10 or 10 or 10%")

    ap.add_argument("--max_total_syn_per_ratio", type=int, default=None)
    ap.add_argument("--max_syn_per_label", type=int, default=None)

    ap.add_argument("--retriever_cache", type=int, default=1)
    ap.add_argument("--st_batch_size", type=int, default=512)
    ap.add_argument("--tmp_root", type=str, default="/content/rag_bug_tmp")

    ap.add_argument("--corpus_comments_cap", type=int, default=300)
    ap.add_argument("--snippet_chars", type=int, default=180)
    ap.add_argument("--dedup_corpus", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--gen_model", type=str, default=MODEL_NAME_DEFAULT)
    ap.add_argument("--load_in_4bit", type=int, default=1)

    ap.add_argument("--m3_emb_model", type=str, default=None,
                    help="Override retriever emb_model for M3 only (default: from config retriever.emb_model).")
    ap.add_argument("--m3_st_device", type=str, default="cpu",
                    help="SentenceTransformer device for M3 embeddings (default: cpu). Use cuda:0 only if you have GPU headroom.")

    args = ap.parse_args()
    logger = setup_logger(args.workdir, args.mode)

    try:
        import yaml
        cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

        cols = cfg["dataset"]["columns"]
        ratios_cfg = cfg["augment"]["syn_ratios"]
        ratios = filter_ratios_from_config(ratios_cfg, args.ratios, logger)

        max_total_syn = int(cfg["augment"].get("max_total_syn_per_ratio", 20000))
        max_syn_per_label = int(cfg["augment"].get("max_syn_per_label", 50))

        if args.max_total_syn_per_ratio is not None:
            max_total_syn = int(args.max_total_syn_per_ratio)
        if args.max_syn_per_label is not None:
            max_syn_per_label = int(args.max_syn_per_label)

        gen_cfg = cfg.get("generator", {})
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 128))
        temperature = float(gen_cfg.get("temperature", 0.7))
        top_p = float(gen_cfg.get("top_p", 0.9))
        max_prompt_tokens = int(gen_cfg.get("max_prompt_tokens", 512))
        repetition_penalty = float(gen_cfg.get("repetition_penalty", 1.1))

        ret_cfg = cfg.get("retriever", {})
        emb_model_cfg = str(ret_cfg.get("emb_model", M3_SAFE_EMB_FALLBACK))
        index_type = str(ret_cfg.get("index_type", "hnsw")).lower()
        snippet_chars = int(ret_cfg.get("snippet_chars", args.snippet_chars))

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {device}")
        if device.startswith("cuda"):
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

        logger.info(f"Caps: max_total_syn_per_ratio={max_total_syn} | max_syn_per_label={max_syn_per_label}")
        logger.info(f"Generator: model={args.gen_model}")
        logger.info(f"Gen params: max_prompt_tokens={max_prompt_tokens} max_new_tokens={max_new_tokens} temp={temperature} top_p={top_p} rep_pen={repetition_penalty}")

        log_memory(logger, "startup")
        log_gpu(logger, "startup")

        # ============================================================
        # Load Mistral generator
        # ============================================================
        logger.info(f"[GEN] Loading tokenizer/model: {args.gen_model}")

        tok = AutoTokenizer.from_pretrained(args.gen_model, use_fast=True, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        device_is_cuda = device.startswith("cuda")
        quant_cfg = None

        if device_is_cuda and bool(args.load_in_4bit):
            try:
                from transformers import BitsAndBytesConfig
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info("[GEN] Using 4-bit quantization via BitsAndBytesConfig")
            except Exception as e:
                quant_cfg = None
                logger.info(f"[GEN] 4-bit quantization not available, fallback to fp16/fp32. Reason: {repr(e)}")

        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch.float16 if device_is_cuda else torch.float32,
        )
        if device_is_cuda:
            model_kwargs["device_map"] = "auto"
        if quant_cfg is not None:
            model_kwargs["quantization_config"] = quant_cfg

        model = AutoModelForCausalLM.from_pretrained(args.gen_model, **model_kwargs)
        model.eval()

        if device_is_cuda:
            torch.set_float32_matmul_precision("high")

        logger.info(f"[GEN] Loaded. dtype={getattr(model, 'dtype', 'unknown')}")
        log_gpu(logger, "after_mistral_load")

        # ============================================================
        # Load clean + train split
        # ============================================================
        clean_path = os.path.join(args.workdir, "clean.json")
        split_path = os.path.join(args.workdir, "splits", "train_ids.json")

        logger.info(f"Loading clean data: {clean_path}")
        df = pd.read_json(clean_path)

        logger.info(f"Loading train ids: {split_path}")
        train_ids = json.load(open(split_path, "r", encoding="utf-8"))
        train_df = df.iloc[train_ids].reset_index(drop=True)

        label_col = cols["label"]
        comments_col = cols["comments"]
        summary_col = cols.get("summary")

        labels = train_df[label_col].astype(str).tolist()
        cnt = Counter(labels)
        minority_labels = pick_minority(cnt)

        logger.info(f"Train rows: {len(train_df)} | unique labels: {len(cnt)} | minority labels: {len(minority_labels)}")
        log_memory(logger, "after_data_load")

        # ============================================================
        # Output dirs (Drive + local tmp)
        # ============================================================
        out_dir_drive = os.path.join(args.workdir, "augmented")
        os.makedirs(out_dir_drive, exist_ok=True)

        out_dir_tmp = os.path.join(args.tmp_root, "augmented_tmp")
        os.makedirs(out_dir_tmp, exist_ok=True)

        rng = np.random.default_rng(args.seed)

        # ============================================================
        # Build retriever (M3 only)
        # ============================================================
        st = index = None
        corpus_texts: Optional[List[str]] = None
        label_to_summaries: Dict[str, List[str]] = {}

        if args.mode == "m3":
            if SentenceTransformer is None or faiss is None:
                raise RuntimeError(f"faiss/sentence-transformers missing: {_IMPORT_ERR}")

            emb_model = args.m3_emb_model if args.m3_emb_model else emb_model_cfg

            if _looks_like_llm_id(emb_model):
                logger.warning(
                    f"[M3] emb_model looks like an LLM ({emb_model}). "
                    f"Falling back to {M3_SAFE_EMB_FALLBACK} for retrieval embeddings."
                )
                emb_model = M3_SAFE_EMB_FALLBACK

            logger.info(f"Retriever: emb_model={emb_model} index_type={index_type} snippet_chars={snippet_chars} st_device={args.m3_st_device}")

            logger.info("[M3] Building M3 corpus: minority-only + short text + optional dedup")
            corpus_df = build_minority_corpus_df(
                train_df=train_df,
                label_col=label_col,
                minority_labels=minority_labels,
                summary_col=summary_col,
                comments_col=comments_col,
                comments_cap=int(args.corpus_comments_cap),
                dedup_by_summary=bool(args.dedup_corpus),
            )
            corpus_texts = corpus_df["_retr_text"].fillna("").astype(str).tolist()
            logger.info(f"[M3] Corpus rows (minority-only): {len(corpus_df)} (from train={len(train_df)})")

            if summary_col and summary_col in corpus_df.columns:
                for lab, grp in corpus_df.groupby(corpus_df[label_col].astype(str)):
                    ss = grp[summary_col].fillna("").astype(str).tolist()
                    ss = [s for s in ss if s and s.strip()]
                    if ss:
                        label_to_summaries[str(lab)] = ss

            k_list = parse_k_list(args.k)
            logger.info(f"[M3] top_k_list={k_list}")
            logger.info(f"[M3] cache={bool(args.retriever_cache)} st_batch_size={args.st_batch_size} tmp_root={args.tmp_root}")

            logger.info(f"[M3] Preparing retriever | corpus={len(corpus_texts)}")
            t0 = time.time()
            st, index, cache_key = build_or_load_retriever(
                corpus_texts=corpus_texts,
                st_device=str(args.m3_st_device),
                emb_model=emb_model,
                index_type=index_type,
                clean_path=clean_path,
                workdir_drive=args.workdir,
                logger=logger,
                cache_enabled=bool(args.retriever_cache),
                st_batch_size=int(args.st_batch_size),
                tmp_root=args.tmp_root,
            )
            logger.info(f"[M3] Retriever ready in {time.time()-t0:.1f}s | cache_key={cache_key}")
            log_memory(logger, "after_retriever")
            log_gpu(logger, "after_retriever")

        # ============================================================
        # Run each ratio
        # ============================================================
        for ratio in ratios:
            ratio_f = float(ratio)
            target = total_syn_needed(len(train_df), ratio_f)
            total_syn = min(target, max_total_syn)

            alloc = allocate_per_label(total_syn, minority_labels, max_syn_per_label)
            planned_total = sum(alloc.values())

            logger.info("=" * 80)
            logger.info(f"Ratio={ratio_f:.2f} target={target} capped_total={total_syn} planned_after_per_label_cap={planned_total}")

            if not alloc:
                logger.info("No allocation (no minority labels or total_syn=0). Skipping ratio.")
                continue

            logger.info(f"Per-label planned: min={min(alloc.values())} max={max(alloc.values())}")

            # -------------------------
            # M2 (UNCHANGED)
            # -------------------------
            if args.mode == "m2":
                tag = f"m2_r{int(ratio_f*100):02d}"
                drive_path = os.path.join(out_dir_drive, f"{tag}.jsonl")
                tmp_path = os.path.join(out_dir_tmp, f"{tag}.jsonl")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

                logger.info(f"[M2] Writing locally to tmp: {tmp_path}")

                for i, (lab, n) in enumerate(alloc.items(), start=1):
                    if n <= 0:
                        continue
                    logger.info(f"[M2] [{i}/{len(alloc)}] Generating n={n} for label={lab}")
                    t0 = time.time()

                    prompt = f"Assigned to {lab}:"
                    gen_texts = fast_generate_batch(
                        model=model, tok=tok, prompt=prompt, num_samples=n,
                        max_prompt_tokens=max_prompt_tokens,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )

                    rows = [
                        {
                            label_col: lab,
                            comments_col: t,
                            "is_synthetic": True,
                            "mode": "m2",
                            "ratio": ratio_f,
                            "k": None,
                            "gen_model": args.gen_model,
                        }
                        for t in gen_texts
                    ]

                    jsonl_write_many(tmp_path, rows)
                    logger.info(f"  wrote {len(rows)} rows | took {time.time()-t0:.1f}s")

                    if i % 25 == 0:
                        log_memory(logger, f"progress i={i}")
                        log_gpu(logger, f"progress i={i}")

                copy_tmp_to_drive(logger, tmp_path, drive_path)
                logger.info(f"[DONE M2 ratio={ratio_f:.2f}] file={drive_path}")
                continue

            # -------------------------
            # M3 (Leakage-guarded)
            # -------------------------
            assert args.mode == "m3"
            assert st is not None and index is not None and corpus_texts is not None

            k_list = parse_k_list(args.k)

            emb_model_used = args.m3_emb_model if args.m3_emb_model else emb_model_cfg
            if _looks_like_llm_id(emb_model_used):
                emb_model_used = M3_SAFE_EMB_FALLBACK

            label_set = set(minority_labels)

            for k_used in k_list:
                tag = f"m3_r{int(ratio_f*100):02d}_k{k_used}"
                drive_path = os.path.join(out_dir_drive, f"{tag}.jsonl")
                tmp_path = os.path.join(out_dir_tmp, f"{tag}.jsonl")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

                logger.info(f"[M3 k={k_used}] Writing locally to tmp: {tmp_path}")

                for i, (lab, n) in enumerate(alloc.items(), start=1):
                    if n <= 0:
                        continue

                    logger.info(f"[M3 k={k_used}] [{i}/{len(alloc)}] Generating n={n} for label={lab}")
                    t0 = time.time()

                    # Leakage-free M3:
                    # - retrieval_query: summary-only (never include label)
                    # - prompt: never contains label; instructs model not to output names/emails
                    summary_seed = choose_summary_for_label(str(lab), label_to_summaries, rng).strip()
                    query_text = summary_seed if summary_seed else ""

                    ctx = retrieve(
                        st=st,
                        index=index,
                        corpus_texts=corpus_texts,
                        query=query_text if query_text else "bug report",
                        k=int(k_used),
                        snippet_chars=int(snippet_chars),
                    )

                    # Sanitize retrieved snippets to remove accidental assignee/name/email leakage
                    ctx = _sanitize_ctx_list(ctx, label_set=label_set)

                    prompt = (
                        "Bug Summary:\n"
                        + (summary_seed if summary_seed else "[no summary provided]") + "\n\n"
                        "Retrieved Context (historical snippets):\n"
                        + "\n---\n".join(ctx) + "\n\n"
                        "Task:\n"
                        "Write a realistic 'Consolidated Comments' section for this bug.\n"
                        "- Do NOT mention any developer name, email address, or assignee.\n"
                        "- Do NOT include 'Assigned to' or 'Assignee' fields.\n"
                        "- Use the context for realism, but paraphrase (do not copy verbatim).\n"
                        "Return ONLY the comments text."
                    )

                    gen_texts = fast_generate_batch(
                        model=model, tok=tok, prompt=prompt, num_samples=n,
                        max_prompt_tokens=max_prompt_tokens,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )

                    # Final leakage scrub on generated outputs (defensive)
                    clean_gen_texts: List[str] = []
                    leak_count = 0
                    for t in gen_texts:
                        stxt = _sanitize_text_no_leak(t, label_set=label_set)
                        if _detect_leak(stxt, label_set=label_set):
                            leak_count += 1
                            continue
                        if stxt:
                            clean_gen_texts.append(stxt)

                    if leak_count > 0:
                        logger.info(f"[M3 k={k_used}] dropped {leak_count}/{len(gen_texts)} generations due to leakage")

                    gen_texts = clean_gen_texts

                    rows = [
                        {
                            label_col: lab,
                            comments_col: t,
                            "is_synthetic": True,
                            "mode": "m3",
                            "ratio": ratio_f,
                            "k": int(k_used),
                            "retrieval_query": query_text,
                            "summary_seed": summary_seed,
                            "leakage_guard": "label_free_query+prompt+sanitize_ctx+sanitize_out+drop_on_detect",
                            "retr_corpus": "minority_only_dedup" if bool(args.dedup_corpus) else "minority_only",
                            "comments_cap": int(args.corpus_comments_cap),
                            "snippet_chars": int(snippet_chars),
                            "emb_model": emb_model_used,
                            "index_type": index_type,
                            "st_device": str(args.m3_st_device),
                            "gen_model": args.gen_model,
                        }
                        for t in gen_texts
                    ]

                    jsonl_write_many(tmp_path, rows)
                    logger.info(f"  wrote {len(rows)} rows | took {time.time()-t0:.1f}s")

                    if i % 25 == 0:
                        log_memory(logger, f"progress i={i}")
                        log_gpu(logger, f"progress i={i}")

                copy_tmp_to_drive(logger, tmp_path, drive_path)
                logger.info(f"[DONE M3 ratio={ratio_f:.2f} k={k_used}] file={drive_path}")

        logger.info("=== ALL DONE ===")

    except Exception as e:
        logger.error("FATAL ERROR")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
