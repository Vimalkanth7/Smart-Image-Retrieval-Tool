# api/main.py

from pathlib import Path
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.db import choose_backend
from src.models import ImageTextEncoder
from src.explain import explain

app = FastAPI(title="Visual Search")
app.mount("/images", StaticFiles(directory="images"), name="images")

backend, store = choose_backend(dim=512, recreate=False)
encoder = ImageTextEncoder()

@app.get("/health")
def health():
    return {"status": "ok", "backend": backend}

@app.get("/", response_class=HTMLResponse)
@app.get("/ui/", response_class=HTMLResponse)
def ui():
    return Path("web/index.html").read_text(encoding="utf-8")

@app.get("/search/text")
def search_text(
    q: str = Query(..., min_length=1),
    top_k: int = 5,
    # mode is REQUIRED, and hybrid is NOT allowed here
    mode: str = Query(..., pattern="^(image|text)$"),
):
    # defensive check (in case someone bypasses the UI)
    if mode not in ("image", "text"):
        raise HTTPException(status_code=400, detail="mode must be 'image' or 'text'")

    qvec = encoder.embed_text(q)
    if mode == "image":
        hits = store.search_vector(qvec, top_k=top_k, vector_name="image_vec")
    else:  # mode == "text"
        hits = store.search_vector(qvec, top_k=top_k, vector_name="text_vec")

    results = []
    for score, payload in hits:
        filename = Path(payload.get("path", "")).name
        results.append({
            # score is still returned by API if you need it for logs;
            # hide it in UI (we already removed it there)
            "score": float(score),
            "image_url": f"/images/{filename}",
            "caption": payload.get("caption", ""),
            "keywords": payload.get("keywords", []),
            "why": explain(q, payload.get("caption", ""), payload.get("keywords", []), []),
        })
    return {"query": q, "results": results}
