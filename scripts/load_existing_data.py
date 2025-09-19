# scripts/load_existing_data.py (replace loader part)
import os, json, argparse, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.db import choose_backend

def load_and_push(data_dir: str):
    iv = np.load(os.path.join(data_dir, "image_vecs.npy"))
    tv = np.load(os.path.join(data_dir, "text_vecs.npy"))

    metas = []
    json_path = os.path.join(data_dir, "meta.json")
    jsonl_path = os.path.join(data_dir, "meta.jsonl")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            metas = json.load(f)
    elif os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                metas.append(json.loads(line))
    else:
        raise FileNotFoundError("meta.json or meta.jsonl not found")

    backend, store = choose_backend(dim=iv.shape[1], recreate=False)
    store.upsert_batch(start_id=0, image_vecs=iv, text_vecs=tv, metas=metas)
    print(f"[DONE] Loaded {len(metas)} points into Qdrant from {data_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="outputs/index")
    args = ap.parse_args()
    load_and_push(args.data_dir)
