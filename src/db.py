import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

_Q_OK = True
try:
    from qdrant_client import QdrantClient, models as qm
except Exception:
    _Q_OK = False


def try_qdrant() -> Optional["QdrantClient"]:
    if not _Q_OK:
        return None
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    try:
        c = QdrantClient(host=host, port=port, timeout=30.0)
        c.get_collections()
        return c
    except Exception as e:
        print(f"[ERROR] Qdrant connection failed ({host}:{port}): {e}")
        return None


def ensure_collection(client: "QdrantClient", name: str, dim: int, recreate: bool = False):
    if recreate:
        client.recreate_collection(
            collection_name=name,
            vectors_config={
                "image_vec": qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
                "text_vec":  qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            },
            hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
        )
    else:
        exists = any(c.name == name for c in client.get_collections().collections)
        if not exists:
            client.create_collection(
                collection_name=name,
                vectors_config={
                    "image_vec": qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
                    "text_vec":  qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
                },
                hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
            )
    # indexes for payload
    for field, schema in (("keywords", qm.PayloadSchemaType.KEYWORD),
                          ("caption",  qm.PayloadSchemaType.TEXT)):
        try:
            client.create_payload_index(collection_name=name, field_name=field, field_schema=schema)
        except Exception:
            pass


class QdrantStore:
    def __init__(self, client: "QdrantClient", collection: str = "photos"):
        self.c = client
        self.col = collection

    def upsert_batch(self, start_id: int, image_vecs: np.ndarray, text_vecs: np.ndarray,
                     metas: List[Dict[str, Any]]):
        points: List[qm.PointStruct] = []
        for i, meta in enumerate(metas):
            points.append(qm.PointStruct(
                id=int(start_id + i),
                vector={"image_vec": image_vecs[i].tolist(), "text_vec": text_vecs[i].tolist()},
                payload=meta
            ))
        self.c.upsert(collection_name=self.col, points=points, wait=True)
        print(f"[QDRANT] upserted {len(points)} points (ids {start_id}..{start_id+len(points)-1})")

    def search_vector(self, q_vec: np.ndarray, top_k: int, vector_name: str):
        hits = self.c.search(
            collection_name=self.col,
            query_vector=qm.NamedVector(name=vector_name, vector=q_vec.astype("float32").tolist()),
            limit=top_k,
            with_payload=True
        )
        return [(float(h.score), h.payload) for h in hits]

    def search_hybrid(self, q_vec: np.ndarray, top_k: int = 5, alpha: float = 0.7, n_candidates: int = 100):
        q = q_vec.astype("float32").tolist()
        img = self.c.search(self.col, query_vector=qm.NamedVector("image_vec", q), limit=n_candidates, with_payload=True)
        txt = self.c.search(self.col, query_vector=qm.NamedVector("text_vec", q),  limit=n_candidates, with_payload=True)
        acc: Dict[int, Dict[str, Any]] = {}
        for h in img:
            pid = int(h.id)
            acc.setdefault(pid, {"score": 0.0, "payload": h.payload})
            acc[pid]["score"] += alpha * float(h.score)
        for h in txt:
            pid = int(h.id)
            acc.setdefault(pid, {"score": 0.0, "payload": h.payload})
            acc[pid]["score"] += (1.0 - alpha) * float(h.score)
        merged = sorted(acc.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        return [(m["score"], m["payload"]) for m in merged]


def choose_backend(dim: int, recreate: bool = False) -> Tuple[str, QdrantStore]:
    cli = try_qdrant()
    if cli is None:
        raise RuntimeError("Qdrant is not reachable at QDRANT_HOST/PORT.")
    ensure_collection(cli, "photos", dim, recreate=recreate)
    return "qdrant", QdrantStore(cli)
