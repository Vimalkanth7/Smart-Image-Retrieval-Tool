Smart Image Retrieval Tool

This project implements an AI-powered visual search system that allows users to search thousands of images using natural language queries.
It uses BLIP for caption generation, OpenCLIP for embeddings, and Qdrant as the vector database.
The backend is served with FastAPI, and a simple web UI enables interactive search.

✨ Features

Automatic caption generation for images.

Text and image embeddings stored in Qdrant for fast semantic search.

Cleaned captions & keywords (removes noisy duplicates like “che che che”).

Search modes:

Image mode: text → image vectors

Text mode: text → caption vectors

Top-K retrieval with explanations.

FastAPI backend + simple web UI for querying.

Dockerized Qdrant for scalable storage.

⚙️ Setup
1. Clone and install
git clone <your-repo>
cd smart_image_retrieval_tool
pip install -r requirements.txt

2. Start Qdrant (vector DB)
docker run -d --name qdrant -p 6333:6333 -v %cd%\qdrant_storage:/qdrant/storage qdrant/qdrant:latest

3. Prepare image data

Download and preprocess images:

python download_images.py --num_images 500


(or copy your own dataset into ./images).

4. Build index (first time only)

Generate captions, embeddings, and save metadata:

python -m scripts.build_index --images_dir ./images --out_dir outputs/index --limit 2500

5. Load existing data (subsequent runs)

If data already exists:

python -m scripts.load_existing_data --data_dir outputs/index

6. Run the API
uvicorn api.main:app --port 8000

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload


🚀 Usage

Open the web UI:

    http://127.0.0.1:8000/ui/


Enter a query (e.g. "ocean", "yellow dog on a couch").

Select search mode (image or text).

View top-5 retrieved images with captions, keywords, and explanations.

📂 Workflow

Image preprocessing → resize + save locally.

Caption generation (BLIP) → clean noisy text, extract keywords.

Embedding generation (OpenCLIP) → create img_vec + text_vec.

Store in Qdrant → vectors + metadata for search.

Search → query is embedded → nearest neighbors fetched.

UI display → shows images, captions, keywords, and reasons.

🛠️ Notes

To re-clean captions (remove noisy words) and rebuild text embeddings:

python -m scripts.clean_meta_and_rebuild_textvecs --data_dir outputs/index --push_qdrant


All vectors and metadata are saved under outputs/index.

Qdrant persists data under qdrant_storage (safe across restarts).

Would you like me to also add a short "Common Issues" section (like port 8000 already in use, or Qdrant not running) to the README? That often helps when others run your repo.



5) What your interviewer will do

Your README should tell them:

git clone https://github.com/<you>/Smart-Image-Retrieval-Tool.git
cd Smart-Image-Retrieval-Tool/docker
docker-compose up --build


Then open:

UI → http://localhost:8000/ui/

API docs → http://localhost:8000/docs

Qdrant dashboard → http://localhost:6333/dashboard

If you included outputs/index_demo, they can search immediately.
If not, they can build vectors (longer):

# inside a local Python env (not needed if you pre-supply demo outputs)
python -m scripts.build_index --images_dir ./images --out_dir outputs/index --limit 2500



1) test 25000

docker stop qdrant
docker rm qdrant
xcopy qdrant_storage qdrant_storage_backup /E /I /H

docker run -d --name qdrant -p 6333:6333 -v %cd%\qdrant_storage:/qdrant/storage qdrant/qdrant:latest




<!-- $env:QDRANT_HOST="127.0.0.1"
$env:QDRANT_PORT="6333"
python scripts\load_existing_data.py --data_dir outputs\index



set QDRANT_HOST=127.0.0.1 && set QDRANT_PORT=6333 && python scripts\load_existing_data.py --data_dir outputs\index
 -->

2nd version


# TO LOAD THE DATA
python -m scripts.load_existing_data --data_dir outputs/index

# TO CLEAN AND REBUILD THE DATA
python scripts\clean_meta_and_rebuild_textvecs.py --data_dir outputs\index --push_qdrant

# ALREADY ran --push_qdrant - Qdrant storage already has the cleaned payload + vectors

docker start qdrant

. .\.venv\Scripts\Activate.ps1
uvicorn api.main:app --host 0.0.0.0 --port 8000

Check:

Qdrant dashboard: http://127.0.0.1:6333/dashboard
 (count ≈ 25,000)

API docs: http://localhost:8000/docs

# did NOT run --push_qdrant yet - only updated files in outputs\index
docker start qdrant
. .\.venv\Scripts\Activate.ps1

python -m scripts.load_existing_data --data_dir outputs/index

uvicorn api.main:app --host 0.0.0.0 --port 8000
