# scripts/verify_artifacts.py
# Quick sanity checks for outputs/index files.
import os, json, numpy as np, argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="outputs/index")
    args = ap.parse_args()

    iv = np.load(os.path.join(args.data_dir, "image_vecs.npy"))
    tv = np.load(os.path.join(args.data_dir, "text_vecs.npy"))
    metas = []
    j = os.path.join(args.data_dir, "meta.json")
    jl = os.path.join(args.data_dir, "meta.jsonl")
    if os.path.exists(j):
        try:
            with open(j, "r", encoding="utf-8") as f:
                metas = json.load(f)
        except json.JSONDecodeError:
            with open(j, "r", encoding="utf-8") as f:
                metas = [json.loads(line) for line in f if line.strip()]
    elif os.path.exists(jl):
        with open(jl, "r", encoding="utf-8") as f:
            metas = [json.loads(line) for line in f if line.strip()]
    else:
        raise SystemExit("No meta.json or meta.jsonl")

    print(f"image_vecs: shape={iv.shape} dtype={iv.dtype}")
    print(f"text_vecs:  shape={tv.shape} dtype={tv.dtype}")
    print(f"meta count:  {len(metas)}")
    print('First meta keys:', sorted(metas[0].keys()) if metas else [])
    if iv.shape[0] != tv.shape[0] or iv.shape[0] != len(metas):
        raise SystemExit("Count mismatch between vectors and meta")
    if iv.shape[1] != 512 or tv.shape[1] != 512:
        raise SystemExit("Vector dimension is not 512")
    print("[OK] Artifacts look consistent.")
