import os, json, hashlib
from typing import Dict, Any, List, Tuple, Iterable
from PIL import Image
import numpy as np
from tqdm import tqdm

from .models import ImageTextEncoder, BlipCaptioner
from .db import QdrantStore


STOPWORDS = {
    "a","an","the","with","in","on","at","and","or","to","from","of","for","over","under","near","by",
    "is","are","be","there","this","that","into","its","it","as","up","down","out","off"
}

def iter_images(folder: str) -> Iterable[str]:
    exts = {".jpg",".jpeg",".png",".webp",".bmp"}
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in exts:
            yield os.path.join(folder, name)

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_keywords(caption: str, top_k: int = 8) -> List[str]:
    tokens = [w.strip(",.?!;:()[]'\"").lower() for w in caption.split()]
    tokens = [w for w in tokens if w.isalpha() and w not in STOPWORDS]
    # keep unique order-preserving, cap at top_k
    uniq = []
    for w in tokens:
        if w not in uniq:
            uniq.append(w)
        if len(uniq) >= top_k:
            break
    return uniq

def preprocess_and_index(
    images_dir: str,
    out_dir: str,
    limit: int | None,
    store: QdrantStore | None,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    os.makedirs(out_dir, exist_ok=True)
    enc = ImageTextEncoder()
    capper = BlipCaptioner()

    img_vecs: List[np.ndarray] = []
    txt_vecs: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []

    paths = list(iter_images(images_dir))
    if limit is not None:
        paths = paths[:limit]

    pbar = tqdm(paths, desc="[Stage] Caption & Embed")
    batch_meta: List[Dict[str, Any]] = []
    batch_img: List[np.ndarray] = []
    batch_txt: List[np.ndarray] = []
    start_id = 0
    for i, path in enumerate(pbar):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue

        caption = capper.caption(img)
        keywords = extract_keywords(caption)
        # text for text_vec = caption + keywords
        text_for_vec = caption + " " + " ".join(keywords)

        ivec = enc.embed_image(img)
        tvec = enc.embed_text(text_for_vec)

        meta = {
            "path": path,
            "caption": caption,
            "keywords": keywords,
            "sha256": sha256_of_file(path),
            "width": img.width,
            "height": img.height,
        }

        img_vecs.append(ivec); txt_vecs.append(tvec); metas.append(meta)
        batch_img.append(ivec); batch_txt.append(tvec); batch_meta.append(meta)

        if store is not None and (len(batch_img) == batch_size or i == len(paths) - 1):
            store.upsert_batch(start_id=start_id, image_vecs=np.vstack(batch_img), text_vecs=np.vstack(batch_txt), metas=batch_meta)
            start_id += len(batch_img)
            batch_img, batch_txt, batch_meta = [], [], []

    img_arr = np.vstack(img_vecs).astype("float32")
    txt_arr = np.vstack(txt_vecs).astype("float32")
    # Save locally
    np.save(os.path.join(out_dir, "image_vecs.npy"), img_arr)
    np.save(os.path.join(out_dir, "text_vecs.npy"), txt_arr)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    return img_arr, txt_arr, metas
