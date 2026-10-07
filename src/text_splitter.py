from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from pdf_loader import load_medical_pdfs


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    # Drop tiny chunks (page numbers, headers) - they only add noise
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 40]

    print(f"Total chunks created: {len(chunks)}")

    return chunks


if __name__ == "__main__":

    print("Starting text splitting...")

    documents = load_medical_pdfs()
    chunks = split_documents(documents)

    if chunks:
        print("\nFirst chunk preview:")
        print(chunks[0].page_content[:500])

        print("\nFirst chunk metadata:")
        print(chunks[0].metadata)