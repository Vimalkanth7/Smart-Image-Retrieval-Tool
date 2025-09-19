"""
Convenience wrapper: preprocess, embed, save locally, and push to Qdrant.
"""
import os
from typing import Optional
from .db import choose_backend
from .preprocess import preprocess_and_index

def build_index(images_dir: str, out_dir: str = "outputs/index", limit: Optional[int] = None, recreate: bool = False):
    backend, store = choose_backend(dim=512, recreate=recreate)
    print(f"[INFO] Vector backend: {backend.upper()}")
    print(f"[INFO] Using images from: {images_dir}")
    print(f"[INFO] Limit: {limit}")
    preprocess_and_index(images_dir=images_dir, out_dir=out_dir, limit=limit, store=store, batch_size=64)
    print(f"[DONE] Indexed {limit or 'all'} images from {images_dir}")

if __name__ == "__main__":
    build_index("./images", "outputs/index", limit=None, recreate=False)
