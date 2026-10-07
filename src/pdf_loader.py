from pathlib import Path

import pymupdf as fitz  # PyMuPDF
from langchain_core.documents import Document

from config import PDF_FOLDER


def load_medical_pdfs(pdf_folder: Path = PDF_FOLDER):
    print("\n==============================")
    print("MEDICAL PDF LOADER")
    print("==============================")

    if not pdf_folder.exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")

    pdf_files = sorted(pdf_folder.glob("*.pdf"))

    print(f"\nPDF files found: {len(pdf_files)}")

    if not pdf_files:
        print("No PDF documents found.")
        return []

    documents = []
    empty_pages = 0

    print("\nLoading PDFs...")

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")

        try:
            with fitz.open(pdf_path) as pdf:
                for page_index, page in enumerate(pdf):
                    text = page.get_text("text").strip()

                    if not text:
                        # Scanned page (image only) - needs OCR
                        empty_pages += 1
                        continue

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                # file name only (not the full path)
                                "source": pdf_path.name,
                                # 1-based page number
                                "page": page_index + 1,
                            },
                        )
                    )

        except Exception as e:
            print(f"Error loading {pdf_path.name}: {e}")

    print(f"\nSuccessfully loaded {len(documents)} pages.")

    if empty_pages:
        print(
            f"WARNING: {empty_pages} pages had no text "
            "(scanned images?). They need OCR."
        )

    return documents


if __name__ == "__main__":

    documents = load_medical_pdfs()

    if documents:
        print("\n==============================")
        print("FIRST DOCUMENT PREVIEW")
        print("==============================")

        print(documents[0].page_content[:1000])

        print("\n==============================")
        print("METADATA")
        print("==============================")

        print(documents[0].metadata)