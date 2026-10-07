import base64
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from gemini_llm import analyze_medical_image
from rag_chain import create_rag_chain


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(page_title="Medical Assistant", page_icon="🩺")

st.title("🩺 Medical Assistant")

st.write(
    "Ask medical questions or upload a medical image for explanation."
)

st.caption(
    "This tool gives general medical information only. "
    "It is not a diagnosis. Please consult a doctor."
)


# -----------------------------------
# LOAD RAG CHAIN (once)
# -----------------------------------

@st.cache_resource
def load_rag_chain():
    return create_rag_chain()


try:
    rag_chain = load_rag_chain()

except Exception as e:
    st.error("Failed to load Medical RAG system.")
    st.exception(e)
    st.stop()


# -----------------------------------
# INPUTS
# -----------------------------------

question = st.text_input("Enter your medical question:")

uploaded_image = st.file_uploader(
    "Upload a medical image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(
        uploaded_image,
        caption="Medical Image",
        use_container_width=True,
    )


# -----------------------------------
# ASK BUTTON
# -----------------------------------

if st.button("Ask"):

    # Nothing provided
    if not question.strip() and uploaded_image is None:

        st.warning("Please enter a question or upload a medical image.")

    # Image question (image + optional text)
    elif uploaded_image is not None:

        with st.spinner("Analyzing medical image..."):

            try:
                image_bytes = uploaded_image.getvalue()

                image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                # png / jpeg - use the real type of the upload
                mime_type = uploaded_image.type or "image/jpeg"

                image_question = question.strip() or (
                    "Please explain this medical image in simple language."
                )

                answer = analyze_medical_image(
                    image_base64,
                    image_question,
                    mime_type,
                )

                st.subheader("🩺 Medical Image Explanation")
                st.write(answer)

            except Exception as e:
                st.error("Unable to analyze the medical image.")
                st.exception(e)

    # Text question -> RAG
    else:

        with st.spinner(
            "Searching medical documents and generating answer..."
        ):

            try:
                answer = rag_chain(question)

                st.subheader("🩺 Answer")
                st.write(answer)

            except Exception as e:
                st.error("Unable to generate an answer.")
                st.exception(e)