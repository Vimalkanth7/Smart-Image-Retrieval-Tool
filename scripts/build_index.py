import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.index import build_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="./images")
    ap.add_argument("--out_dir", default="outputs/index")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--recreate", action="store_true", help="Drop & recreate Qdrant collection")
    args = ap.parse_args()

    build_index(images_dir=args.images_dir, out_dir=args.out_dir, limit=args.limit, recreate=args.recreate)
