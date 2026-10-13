"""Build the Chroma vector store from the PDFs.

    python src/vector_store.py            # fresh build (deletes old DB)
    python src/vector_store.py --resume   # continue an interrupted build
"""

import argparse
import hashlib
import shutil
import sys

from langchain_chroma import Chroma

from config import EMBED_BATCH_SIZE, VECTORSTORE_PATH
from embeddings import get_embedding_model
from pdf_loader import load_medical_pdfs
from text_splitter import split_documents


def chunk_id(chunk):
    """Stable ID: same chunk -> same ID (enables --resume, removes duplicates)."""
    key = (
        f"{chunk.metadata.get('source')}|"
        f"{chunk.metadata.get('page')}|"
        f"{chunk.page_content}"
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def create_vector_store(resume=False):

    print("Loading medical documents...")
    documents = load_medical_pdfs()

    if not documents:
        raise RuntimeError("No text could be loaded from the PDFs. Check data/medical_pdfs.")

    print("Splitting documents...")
    chunks = split_documents(documents)

    unique = {}
    for chunk in chunks:
        unique.setdefault(chunk_id(chunk), chunk)

    ids = list(unique)
    total = len(ids)
    print(f"Unique chunks: {total}")

    embedding_model = get_embedding_model()

    if VECTORSTORE_PATH.exists() and not resume:
        print(f"Removing old vector store: {VECTORSTORE_PATH}")
        try:
            shutil.rmtree(VECTORSTORE_PATH)
        except PermissionError:
            raise RuntimeError(
                "Cannot delete the old vector store (file in use). "
                "Close other Python/Streamlit/uvicorn windows, pause "
                "OneDrive sync, then run again."
            )

    VECTORSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        persist_directory=str(VECTORSTORE_PATH),
        embedding_function=embedding_model,
    )

    already_done = set()
    if resume:
        already_done = set(vector_store.get(include=["metadatas"])["ids"])
        print(f"Resuming: {len(already_done)} chunks already stored.")

    todo = [i for i in ids if i not in already_done]
    print(f"Embedding {len(todo)} chunks...")

    stored = len(already_done)

    try:
        for start in range(0, len(todo), EMBED_BATCH_SIZE):
            batch_ids = todo[start:start + EMBED_BATCH_SIZE]
            batch = [unique[i] for i in batch_ids]

            vector_store.add_documents(batch, ids=batch_ids)

            stored += len(batch_ids)
            print(f"Stored {stored}/{total} chunks")

    except KeyboardInterrupt:
        print(f"\nStopped at {stored}/{total}. Continue with: python src/vector_store.py --resume")
        sys.exit(1)

    except Exception as e:
        print(f"\nStopped at {stored}/{total}: {e}")
        print("Fix the reported issue, then continue with: python src/vector_store.py --resume")
        sys.exit(1)

    print("ChromaDB vector store created successfully!")
    print(f"Stored at: {VECTORSTORE_PATH}")
    print(f"Total chunks stored: {stored}")
    return vector_store


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted build")
    create_vector_store(resume=parser.parse_args().resume)