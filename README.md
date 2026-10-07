# 🩺 Medical RAG Assistant

A multilingual medical information assistant built with **LangChain**, **ChromaDB**, **HuggingFace embeddings** and **Google Gemini**.
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
| Streamlit web app | `src/app.py` |

---

## 📁 Project structure

```
medical_rag/
├── .env                  # your GEMINI_API_KEY (never commit this)
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── medical_pdfs/     # put your PDF files here
├── vectorstore/
│   └── chroma_db/        # created by vector_store.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── app.py
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

**5. Build the vector database** (one time, takes a few minutes on CPU)

```bash
python src/vector_store.py
```

**6. Run the app**

```bash
streamlit run src/app.py
```

Open http://localhost:8501.

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

If you change the embedding model or chunk settings, **rebuild the vector store** with `python src/vector_store.py`.

---

## 🧠 Design notes (what I learned)

- **Chunk size must match the embedding model.** `paraphrase-multilingual-MiniLM-L12-v2` reads only the first 128 tokens, so 1000-character chunks were partly never embedded. Chunks are now ~500 characters.
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

Python · LangChain · ChromaDB · sentence-transformers · Google Gemini · PyMuPDF · Streamlit

---

## 📄 License

Add your preferred license (for example MIT). Medical PDFs are **not** included, and each document remains under its original license.
