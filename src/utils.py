import os
import json
import random
import re
from typing import Any, Dict, List, Union, Optional

import numpy as np
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _parse_json_stream(text: str) -> List[Any]:
    """
    Parses a file that contains multiple JSON values one after another.
    Works even if each JSON object spans multiple lines (pretty printed).

    Example supported formats:
      - {}{}{}
      - {}\n{}\n{}
      - { ... }\n{ ... }\n  (multi-line objects)
    """
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    out: List[Any] = []

    while True:
        # skip whitespace
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break

        try:
            obj, j = dec.raw_decode(text, i)
        except json.JSONDecodeError as e:
            # Helpful context for debugging corrupted files
            start = max(0, i - 120)
            end = min(n, i + 120)
            snippet = text[start:end].replace("\n", "\\n")
            raise json.JSONDecodeError(
                f"{e.msg} (while stream-parsing). Around: ...{snippet}...",
                e.doc,
                e.pos,
            ) from None

        out.append(obj)
        i = j

    return out


def read_json(path: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Robust JSON reader that supports:
      1) Standard JSON (single object/array)
      2) JSONL / NDJSON (one JSON object per line)
      3) Concatenated / multi-line JSON objects (stream of JSON values)

    Returns:
      - dict / list if the file is a single JSON value
      - list of objects if the file contains many JSON values
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    text = (text or "").strip()
    if not text:
        return []

    # Case A: looks like a single JSON value (array/object)
    # Try normal parse first.
    if text[0] in ("{", "["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If this fails, it might be concatenated JSON objects (or corrupted).
            # Fall through to stream parse.
            pass

    # Case B: Try classic JSONL (one object per line).
    # This is fast and covers the common correct JSONL case.
    try:
        rows: List[Any] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        # If we got at least one row and parsing succeeded, return it.
        if rows:
            return rows
    except json.JSONDecodeError:
        # Not strict JSONL (maybe multi-line objects). Fall through.
        pass

    # Case C: Stream parse concatenated JSON values (supports multi-line objects).
    return _parse_json_stream(text)


def write_json(path: str, obj: Any, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def write_jsonl(path: str, rows: List[Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s
