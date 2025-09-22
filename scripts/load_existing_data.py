# scripts/load_existing_data.py
import os, json, argparse, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.db import choose_backend


def load_meta_file(data_dir: str):
    """Load metadata from meta.json or meta.jsonl (array or JSONL style)."""
    json_path = os.path.join(data_dir, "meta.json")
    jsonl_path = os.path.join(data_dir, "meta.jsonl")

    def load_json_array(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_jsonl(path: str):
        metas = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metas.append(json.loads(line))
        return metas

    # Case 1: meta.json exists
    if os.path.exists(json_path):
        try:
            data = load_json_array(json_path)
            if isinstance(data, list):
                return data
            # fallback: treat as JSONL if not a list
            return load_jsonl(json_path)
        except json.JSONDecodeError:
            return load_jsonl(json_path)

    # Case 2: meta.jsonl exists
    if os.path.exists(jsonl_path):
        return load_jsonl(jsonl_path)

    raise FileNotFoundError("meta.json or meta.jsonl not found in {data_dir}")


def load_and_push(data_dir: str, recreate: bool = True):
    iv = np.load(os.path.join(data_dir, "image_vecs.npy"))
    tv = np.load(os.path.join(data_dir, "text_vecs.npy"))

    metas = load_meta_file(data_dir)

    backend, store = choose_backend(dim=iv.shape[1], recreate=False)
    store.upsert_batch(start_id=0, image_vecs=iv, text_vecs=tv, metas=metas)

    print(f"[DONE] Loaded {len(metas)} points into Qdrant from {data_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="outputs/index")
    ap.add_argument("--no_recreate", action="store_true",
                    help="Do not recreate collection, just upsert into existing one")
    args = ap.parse_args()

    load_and_push(args.data_dir, recreate=not args.no_recreate)
