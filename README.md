# 🩺 Medical RAG Assistant

A multilingual medical information assistant built with **LangChain**, **ChromaDB**, local **FastEmbed** embeddings and **Google Gemini**.
Ask questions in **English, Telugu (script) or Telugu written in English letters (Tenglish)** and get answers grounded in medical PDFs (WHO manuals, clinical guidelines, first-aid manuals, drug lists). You can also upload a medical image (medicine strip, lab report) for a plain-language explanation.

> ⚠️ **Disclaimer:** This project gives general medical information only. It is **not** a diagnosis or medical advice. Always consult a qualified doctor or pharmacist.

---

## ✨ Features

- **RAG over your own PDFs:** answers come from the documents, with source file and page numbers.
- **Multilingual:** English, Telugu script and Tenglish questions. Telugu-script questions are translated to English for retrieval, and the answer is written back in the user's language style.
- **Hybrid retrieval:** vector similarity + keyword scoring, query expansion, de-duplication, and a penalty for bibliography / reference pages.
- **Brand → generic medicine mapping:** e.g. *Dolo 650* → *paracetamol* so brand-name questions still find the right content.
- **Image explanation:** upload a medicine strip or lab report and get a simple explanation (Gemini vision).
- **Safety rules in the prompt:** no definitive diagnosis, no prescriptions, emergency advice when needed.

---

## 🧱 How it works

```
PDFs ──► pdf_loader ──► text_splitter ──► embeddings ──► ChromaDB   (build once)

Question ──► (translate if Telugu script) ──► query expansion
         ──► vector search (ChromaDB) ──► keyword re-ranking + dedup
         ──► top chunks + question ──► Gemini ──► answer + sources
```

| Step | File |
|------|------|
| Settings (paths, chunk size, top-k) | `src/config.py` |
| Load PDF text | `src/pdf_loader.py` |
| Split into chunks | `src/text_splitter.py` |
| Embedding model | `src/embeddings.py` |
| Build the vector database | `src/vector_store.py` |
| Brand → generic medicine names | `src/medicine_mapping.py` |
| Search, re-rank, dedupe | `src/retriever.py` |
| Prompt template | `src/prompt_template.py` |
| Gemini LLM + image analysis | `src/gemini_llm.py` |
| RAG pipeline | `src/rag_chain.py` |
| FastAPI backend (REST API) | `src/api.py` |
| HTML/CSS/JavaScript web frontend | `src/static/` |
| Optional Streamlit frontend (calls the API) | `src/app.py` |
| API tests | `tests/test_api.py` |

---

## 📁 Project structure

```
medical_rag/
├── .env                  # your GEMINI_API_KEY (never commit this)
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt  # pytest + httpx for the tests
├── README.md
├── data/
│   └── medical_pdfs/     # put your PDF files here
├── vectorstore/
│   └── chroma_db/        # created by vector_store.py
├── tests/
│   └── test_api.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── api.py            # FastAPI backend
    ├── app.py            # Optional Streamlit frontend
    ├── static/           # HTML, CSS and JavaScript web frontend
    ├── rag_chain.py
    ├── retriever.py
    ├── embeddings.py
    ├── vector_store.py
    ├── pdf_loader.py
    ├── text_splitter.py
    ├── gemini_llm.py
    ├── prompt_template.py
    └── medicine_mapping.py
```

---

## 🚀 Setup

**1. Clone and create a virtual environment** (Python 3.11 or 3.12 recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add your Gemini API key**

Get a key from [Google AI Studio](https://aistudio.google.com/), then create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
# Optional: choose a different Gemini model
# GEMINI_MODEL=gemini-2.5-flash-lite
# Optional: hide retrieval debug prints
# RAG_DEBUG=0
```

**4. Add your PDFs**

Put your medical PDF files in `data/medical_pdfs/`.
(PDFs are not included in this repository. Use freely available sources such as WHO publications, and respect each document's license.)

**5. Build the vector database** (one-time local CPU embedding job)

```bash
python src/vector_store.py
```

The first run downloads the multilingual ONNX model. It is cached locally
under `.cache/fastembed` by default. Embedding generation runs locally and does
not use Gemini's embedding API quota. Gemini is still used to generate answers
and analyze uploaded images.

If a build is interrupted, continue it with:

```bash
python src/vector_store.py --resume
```

The vector database uses 384-dimensional vectors. The previous Gemini
embedding database is incompatible and must be rebuilt once with
`python src/vector_store.py`.

### Render storage

Render's filesystem is ephemeral unless the service has a persistent disk
(available on paid services). For a Render service with a persistent disk:

1. Attach a disk mounted at `/var/data`.
2. Set `EMBEDDING_CACHE_PATH=/var/data/fastembed` and
   `VECTORSTORE_PATH=/var/data/chroma_db` in the service environment.
3. Keep the Render build command to dependency installation; do not build the
   vector database in the build phase. After the app deploys, run
   `python src/vector_store.py` from the service shell to create the index on
   the mounted disk. The model download is cached there too, so it is not
   downloaded again after restarts.
4. Keep the disk attached for future deploys. Do not include `vectorstore/`
   or `.cache/` in Git.

The FastEmbed model download is approximately 220 MB; the 384-dimensional
index is substantially smaller than an index built with 768-dimensional
vectors. Leave room on the disk for the model cache, index metadata, and
future growth. On deployments without persistent storage, the local model
cache and vector index are lost after a restart or redeploy and must be
downloaded/rebuilt again.

**6. Start the backend** (terminal 1, from the project root)

```bash
uvicorn api:app --reload --app-dir src
```

API docs (Swagger UI): http://127.0.0.1:8000/docs

**7. Open the web frontend**

Visit http://127.0.0.1:8000 in your browser. The FastAPI server hosts the HTML/CSS/JavaScript chat UI, including image upload and backend status.

Alternatively, start the Streamlit UI in a second terminal:

```bash
streamlit run src/app.py
```

Open http://localhost:8501. To point the Streamlit UI at another backend URL, set the `API_URL` environment variable.

---

## 🔌 REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status and whether the RAG chain and vector store are available |
| `POST` | `/ask` | JSON `{"question": "..."}` → `{"answer": "..."}` |
| `POST` | `/analyze-image` | Multipart form: `image` (PNG/JPEG, max 5 MB) + optional `question` → `{"answer": "..."}` |

Example:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "hypertension lakshanalu enti?"}'
```

Status codes: `422` invalid input, `415` unsupported image type, `413` image too large, `503` RAG not ready, `500` internal error.

Run the API tests (they use fakes, so no Gemini key or vector store is needed):

```bash
pip install -r requirements-dev.txt
pytest
```

---

## 🧪 Test each part separately

```bash
python src/embeddings.py        # prints "Embedding dimension: 384"
python src/pdf_loader.py        # loads PDFs
python src/text_splitter.py     # shows chunk count
python src/gemini_llm.py        # tests Gemini key
python src/retriever.py         # shows retrieved chunks
python src/rag_chain.py         # full RAG answer
```

---

## 💬 Example questions

| Type | Example |
|------|---------|
| English | `What is hypertension?` |
| Tenglish | `hypertension lakshanalu enti?` |
| Telugu script | `హైపర్‌టెన్షన్ అంటే ఏమిటి?` |
| Medicine | `Dolo 650 deniki use chestaru?` |
| Image | Upload a medicine strip or lab report |

---

## ⚙️ Configuration

All main settings are in `src/config.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 100 | Chunk size in characters |
| `TOP_K_PER_QUERY` | 6 | Results fetched per expanded query |
| `FINAL_TOP_K` | 6 | Chunks sent to Gemini |
| `KEYWORD_WEIGHT`, `KEYWORD_CAP` | 0.01, 40 | Strength of keyword re-ranking |
| `VECTORSTORE_PATH` | `vectorstore/chroma_db` | Chroma database directory; point to the Render disk |
| `EMBEDDING_CACHE_PATH` | `.cache/fastembed` | Local model cache; point to the Render disk |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | FastEmbed multilingual ONNX model |

If you change the embedding model or chunk settings, **rebuild the vector store** with `python src/vector_store.py`.

---

## 🧠 Design notes (what I learned)

- **Chunk size must match the embedding model.** Keep configured chunks within the model's 128-token input limit to avoid truncating content during embedding.
- **Distance vs keyword score scale.** Chroma returns a small distance; adding large raw keyword scores made keywords dominate the ranking. The keyword bonus is now capped and scaled.
- **Bibliography pages pollute results.** Reference sections repeat topic words ("diabetes", "insulin") but contain no useful content, so they are penalised.
- **Duplicate PDFs waste retrieval slots.** Duplicate files are removed and chunks are de-duplicated by content.
- **Rebuilding must start clean.** `Chroma.from_documents` appends to an existing database, so the store is deleted before each rebuild.

---

## ⚠️ Limitations

- Answers are only as good as the PDFs provided. Topics missing from the documents will not be answered well.
- Scanned (image-only) PDF pages are skipped (no OCR yet).
- Tenglish understanding relies on a small set of keywords plus the multilingual embedding model.
- LLM output can still contain mistakes. This is not a substitute for professional medical advice.

---

## 🔮 Possible improvements

- OCR for scanned PDFs
- Evaluation set with retrieval hit-rate metrics
- Cross-encoder re-ranking
- Chat history / follow-up questions
- Larger brand-to-generic medicine mapping
- Deployment (Streamlit Community Cloud / Hugging Face Spaces)

---

## 🛠️ Tech stack

Python · FastAPI · LangChain · ChromaDB · FastEmbed · ONNX Runtime · Google Gemini · PyMuPDF · Streamlit · pytest

---

## 📄 License

Add your preferred license (for example MIT). Medical PDFs are **not** included, and each document remains under its original license.
