import shutil

from langchain_chroma import Chroma

from config import VECTORSTORE_PATH
from embeddings import get_embedding_model
from pdf_loader import load_medical_pdfs
from text_splitter import split_documents

BATCH_SIZE = 500


def create_vector_store():

    print("Loading medical documents...")
    documents = load_medical_pdfs()

    if not documents:
        raise RuntimeError(
            "No text could be loaded from the PDFs. "
            "Check data/medical_pdfs."
        )

    print("Splitting documents...")
    chunks = split_documents(documents)

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    # Adding to an existing DB would store every chunk twice,
    # so always start from a clean folder.
    if VECTORSTORE_PATH.exists():
        print(f"Removing old vector store: {VECTORSTORE_PATH}")
        try:
            shutil.rmtree(VECTORSTORE_PATH)
        except PermissionError:
            raise RuntimeError(
                "Cannot delete the old vector store (file in use). "
                "Close other Python/Streamlit windows, pause OneDrive "
                "sync, then run again."
            )

    VECTORSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Creating ChromaDB vector store...")

    vector_store = Chroma(
        persist_directory=str(VECTORSTORE_PATH),
        embedding_function=embedding_model,
    )

    total = len(chunks)

    # Add in batches so we can see progress
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        vector_store.add_documents(batch)
        print(f"Embedded {min(start + BATCH_SIZE, total)}/{total} chunks")

    print("ChromaDB vector store created successfully!")
    print(f"Stored at: {VECTORSTORE_PATH}")
    print(f"Total chunks stored: {total}")

    return vector_store


if __name__ == "__main__":
    create_vector_store()