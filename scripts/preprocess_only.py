# scripts/preprocess_only.py
# Build artifacts WITHOUT Qdrant:
#   outputs/index/image_vecs.npy
#   outputs/index/text_vecs.npy
#   outputs/index/meta.json  (JSONL content; loader handles it)
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import preprocess_and_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="./images")
    ap.add_argument("--out_dir", default="outputs/index")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    # store=None => do NOT push to Qdrant; only write artifacts to disk
    preprocess_and_index(
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        limit=args.limit,
        store=None,
        batch_size=args.batch_size,
    )
    print("[DONE] Wrote image_vecs.npy, text_vecs.npy, meta.json under", args.out_dir)
