import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# per-model files:
# m1__LinearSVC.json
# m2__CNN.json
# m3_k5__LinearSVC.json
PER_MODEL_RE = re.compile(
    r"^(m[123])(?:_k(\d+))?__([A-Za-z0-9_.-]+)\.json$",
    re.IGNORECASE
)

def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Dict[str, Any], p: Path) -> None:
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def rebuild_blob(files: List[Path], mode: str, k: Optional[int]) -> Dict[str, Any]:
    by_model: Dict[str, List[Path]] = {}
    for p in files:
        m = PER_MODEL_RE.match(p.name)
        model = m.group(3)
        by_model.setdefault(model, []).append(p)

    models = {}
    n_train = n_test = n_classes_train = None

    for model, paths in sorted(by_model.items()):
        latest = max(paths, key=lambda x: x.stat().st_mtime)
        obj = load_json(latest)

        # model JSON may already be nested
        if "models" in obj and model in obj["models"]:
            models[model] = obj["models"][model]
        else:
            models[model] = obj

        n_train = n_train or obj.get("n_train")
        n_test = n_test or obj.get("n_test")
        n_classes_train = n_classes_train or obj.get("n_classes_train")

    return {
        "mode": mode,
        "k": k,
        "n_train": n_train,
        "n_test": n_test,
        "n_classes_train": n_classes_train,
        "models": models,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_path", required=True, help="Absolute or relative path to metrics folder")
    ap.add_argument("--recursive", action="store_true", help="Scan subfolders")
    ap.add_argument("--write", action="store_true", help="Write __ALL.json files")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_path).resolve()
    if not metrics_dir.is_dir():
        raise FileNotFoundError(f"Invalid path: {metrics_dir}")

    files = metrics_dir.rglob("*.json") if args.recursive else metrics_dir.glob("*.json")

    groups: Dict[Tuple[str, Optional[int]], List[Path]] = {}

    for p in files:
        if "__ALL.json" in p.name:
            continue
        m = PER_MODEL_RE.match(p.name)
        if not m:
            continue
        mode = m.group(1).lower()
        k = int(m.group(2)) if m.group(2) else None
        groups.setdefault((mode, k), []).append(p)

    if not groups:
        raise RuntimeError("No per-model JSON files found.")

    for (mode, k), paths in sorted(groups.items()):
        blob = rebuild_blob(paths, mode, k)
        out_name = f"{mode}_k{k}__ALL.json" if (mode == "m3" and k is not None) else f"{mode}__ALL.json"
        out_path = metrics_dir / out_name

        print(f"[BUILD] {out_path.name}  | models={len(blob['models'])}")

        if args.write:
            save_json(blob, out_path)

    if args.write:
        print("\n✅ Regeneration completed.")
    else:
        print("\n(Dry run) Add --write to generate files.")

if __name__ == "__main__":
    main()