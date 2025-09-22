# scripts/clean_meta_and_rebuild_textvecs.py
import os, sys, json, argparse
from typing import List, Dict, Any
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tqdm import tqdm

from src.models import ImageTextEncoder
from src.clean import clean_caption_and_keywords
from src.db import choose_backend


def _load_meta(data_dir: str) -> list[dict]:
    json_path = os.path.join(data_dir, "meta.json")
    jsonl_path = os.path.join(data_dir, "meta.jsonl")

    def load_json_array(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_jsonl(path: str):
        metas = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                metas.append(json.loads(s))
        return metas

    if os.path.exists(json_path):
        try:
            data = load_json_array(json_path)
            if isinstance(data, list):
                return data
            return load_jsonl(json_path)  # fallback if it was JSONL content
        except json.JSONDecodeError:
            return load_jsonl(json_path)

    if os.path.exists(jsonl_path):
        return load_jsonl(jsonl_path)

    raise FileNotFoundError(f"No meta.json or meta.jsonl found in {data_dir}")


def _save_meta(data_dir: str, metas: List[Dict[str, Any]], use_json: bool = True):
    if use_json:
        outp = os.path.join(data_dir, "meta.json")
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(metas, f, ensure_ascii=False, indent=2)
    else:
        outp = os.path.join(data_dir, "meta.jsonl")
        with open(outp, "w", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")


def rebuild_text_vecs(
    data_dir: str,
    overwrite: bool = True,
    push_qdrant: bool = False,
    batch: int = 256,
):
    # 1) load existing meta + (keep) image_vecs
    metas = _load_meta(data_dir)
    iv_path = os.path.join(data_dir, "image_vecs.npy")
    if not os.path.exists(iv_path):
        raise FileNotFoundError("image_vecs.npy not found; needed for dimension consistency")
    image_vecs = np.load(iv_path)
    dim = image_vecs.shape[1]

    # 2) clean captions & rebuild keywords
    updated: List[Dict[str, Any]] = []
    texts: List[str] = []
    for m in metas:
        raw = m.get("caption", "") or ""
        clean_cap, kws = clean_caption_and_keywords(raw, keyword_top_k=8)
        m["caption"] = clean_cap
        m["keywords"] = kws
        updated.append(m)
        texts.append(clean_cap + " " + " ".join(kws))

    # 3) re-embed text using OpenCLIP (with progress bar)
    enc = ImageTextEncoder()
    text_vecs = np.zeros((len(texts), dim), dtype="float32")

    pbar = tqdm(total=len(texts), desc="[EMBED TEXT]", unit="vec")
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        for j, t in enumerate(chunk):
            text_vecs[i + j] = enc.embed_text(t)
        pbar.update(len(chunk))
    pbar.close()

    # 4) save updated meta + text_vecs
    _save_meta(data_dir, updated, use_json=True)
    tv_path = os.path.join(data_dir, "text_vecs.npy")
    if overwrite:
        np.save(tv_path, text_vecs)
        print(f"[CLEAN] Overwrote {tv_path} with cleaned text vectors.")
    else:
        out_alt = os.path.join(data_dir, "text_vecs.cleaned.npy")
        np.save(out_alt, text_vecs)
        print(f"[CLEAN] Wrote {out_alt}")

    # 5) optional: push to Qdrant (IDs preserved: 0..N-1) with progress bar
    if push_qdrant:
        backend, store = choose_backend(dim=dim, recreate=False)
        batch_size = 256
        pushed = 0
        pbar_q = tqdm(total=len(updated), desc="[QDRANT UPSERT]", unit="pt")
        for i in range(0, len(updated), batch_size):
            store.upsert_batch(
                start_id=i,
                image_vecs=image_vecs[i:i + batch_size],
                text_vecs=text_vecs[i:i + batch_size],
                metas=updated[i:i + batch_size],
            )
            chunk_n = len(updated[i:i + batch_size])
            pushed += chunk_n
            pbar_q.update(chunk_n)
        pbar_q.close()
        print(f"[QDRANT] Finished upserting {pushed} cleaned points")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="outputs/index")
    ap.add_argument("--no_overwrite", action="store_true",
                    help="Write text_vecs.cleaned.npy instead of overwriting")
    ap.add_argument("--push_qdrant", action="store_true",
                    help="Upsert cleaned payload + vectors to Qdrant")
    ap.add_argument("--batch", type=int, default=256,
                    help="Embedding batch size (CPU/GPU dependent)")
    args = ap.parse_args()

    rebuild_text_vecs(
        data_dir=args.data_dir,
        overwrite=not args.no_overwrite,
        push_qdrant=args.push_qdrant,
        batch=args.batch,
    )
