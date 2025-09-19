# Smart Image Retrieval Tool

This project implements an **AI-powered Image search Retrieval system** that allows users to search thousands of images to retrieve the relevant top 5 images
It uses **BLIP** for caption generation, **OpenCLIP** for embeddings, and **Qdrant** as the vector database.  
The backend is served with **FastAPI**, and a simple **web UI** enables interactive search.

---

## ✨ Features
- Automatic **caption generation** for images.  
- **Text and image embeddings** stored in Qdrant for fast semantic search.  
- **Cleaned captions & keywords** (removes noisy duplicates like *“che che che”*).  
- **Search modes**:  
  - *Image mode*: text → image vectors  
  - *Text mode*: text → caption vectors  
- **Top-K retrieval** with explanations.  
- **FastAPI backend + simple web UI** for querying.  
- **Dockerized Qdrant** for scalable storage.

---

## ⚙️ Setup

### 1. Clone and install
```bash
git clone <your-repo>
cd smart_image_retrieval_tool
pip install -r requirements.txt
```

### 2. Start Qdrant (vector DB)
```bash
docker run -d --name qdrant -p 6333:6333 -v %cd%\qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

### 3. Prepare image data
Download and preprocess images:
```bash
python download_images.py --num_images 500
```
(or copy your own dataset into `./images`).

### 4. Build index (first time only)
Generate captions, embeddings, and save metadata:
```bash
python -m scripts.build_index --images_dir ./images --out_dir outputs/index --limit 2500
```

### 5. Load existing data (subsequent runs)
If data already exists:
```bash
python -m scripts.load_existing_data --data_dir outputs/index
```

### 6. Run the API
```bash
uvicorn api.main:app --port 8000
```

---

## 🚀 Usage

- Open the web UI:
  ```
  http://127.0.0.1:8000/ui/
  ```
- Enter a query (e.g. *"ocean"*, *"yellow dog on a couch"*).  
- Select search mode (*image* or *text*).  
- View top-5 retrieved images with captions, keywords, and explanations.  

---

## 📂 Workflow

1. **Image preprocessing** → resize + save locally.  
2. **Caption generation (BLIP)** → clean noisy text, extract keywords.  
3. **Embedding generation (OpenCLIP)** → create `img_vec` + `text_vec`.  
4. **Store in Qdrant** → vectors + metadata for search.  
5. **Search** → query is embedded → nearest neighbors fetched.  
6. **UI display** → shows images, captions, keywords, and reasons.

---

## 🛠️ Notes
- To re-clean captions (remove noisy words) and rebuild text embeddings:
  ```bash
  python -m scripts.clean_meta_and_rebuild_textvecs --data_dir outputs/index --push_qdrant
  ```
- All vectors and metadata are saved under `outputs/index`.  
- Qdrant persists data under `qdrant_storage` (safe across restarts).


### 🚀 Quick Start (Docker)

Clone repo and run:

```bash
git clone https://github.com/Vimalkanth7/Smart-Image-Retrieval-Tool.git
cd Smart-Image-Retrieval-Tool/docker
docker-compose up --build

