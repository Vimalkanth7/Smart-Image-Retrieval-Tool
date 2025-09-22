# scripts/build_index_offline.py
import argparse, os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.preprocess import preprocess_and_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="./images", help="Folder containing downloaded images")
    ap.add_argument("--limit", type=int, default=None, help="Max images to process (None = all)")
    ap.add_argument("--out_dir", default="outputs/index", help="Where to save .npy/meta.json")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    if not os.path.isdir(args.images_dir):
        sys.exit(f"[ERROR] images_dir not found: {args.images_dir}")
    if len([f for f in os.listdir(args.images_dir) if os.path.isfile(os.path.join(args.images_dir, f))]) == 0:
        sys.exit(f"[ERROR] images_dir is empty: {args.images_dir}")

    print(f"[INFO] Running OFFLINE build (no Qdrant).")
    print(f"[INFO] Images: {args.images_dir}")
    print(f"[INFO] Limit: {args.limit}")
    print(f"[INFO] Saving to: {args.out_dir}")

    n = preprocess_and_index(
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        limit=args.limit,
        store=None,
        batch_size=args.batch_size,
    )
    # print(f"[DONE] Indexed {n} images offline → {args.out_dir}")
