"""Streamlit frontend. It does NOT run the RAG pipeline itself:
it calls the FastAPI backend (src/api.py) over HTTP.

Start the backend first:   uvicorn api:app --reload --app-dir src
Then start this UI:        streamlit run src/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = 180  # first request can be slow


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
# BACKEND HELPERS
# -----------------------------------

def backend_status():
    """Return (online, rag_ready)."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        data = response.json()
        return True, bool(data.get("rag_ready"))
    except Exception:
        return False, False


def error_message(response):
    """Readable error text from a failed API response."""
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def ask_backend(question):
    response = requests.post(
        f"{API_URL}/ask",
        json={"question": question},
        timeout=TIMEOUT_SECONDS,
    )

    if response.status_code != 200:
        raise RuntimeError(error_message(response))

    return response.json()["answer"]


def analyze_image_backend(uploaded_image, question):
    response = requests.post(
        f"{API_URL}/analyze-image",
        files={
            "image": (
                uploaded_image.name,
                uploaded_image.getvalue(),
                uploaded_image.type or "image/jpeg",
            )
        },
        data={"question": question},
        timeout=TIMEOUT_SECONDS,
    )

    if response.status_code != 200:
        raise RuntimeError(error_message(response))

    return response.json()["answer"]


# -----------------------------------
# SIDEBAR: BACKEND STATUS
# -----------------------------------

online, ready = backend_status()

with st.sidebar:
    st.subheader("Backend")

    if online and ready:
        st.success("Online")
    elif online:
        st.warning("Starting up...")
    else:
        st.error("Offline")

    st.caption(f"API: {API_URL}")

if not online:
    st.error(
        "Cannot reach the backend. Start it in another terminal:\n\n"
        "`uvicorn api:app --reload --app-dir src`"
    )
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
                answer = analyze_image_backend(uploaded_image, question.strip())

                st.subheader("🩺 Medical Image Explanation")
                st.write(answer)

            except requests.exceptions.RequestException as e:
                st.error("Could not reach the backend.")
                st.exception(e)

            except Exception as e:
                st.error(f"Unable to analyze the medical image: {e}")

    # Text question -> RAG
    else:

        with st.spinner(
            "Searching medical documents and generating answer..."
        ):

            try:
                answer = ask_backend(question.strip())

                st.subheader("🩺 Answer")
                st.write(answer)

            except requests.exceptions.RequestException as e:
                st.error("Could not reach the backend.")
                st.exception(e)

            except Exception as e:
                st.error(f"Unable to generate an answer: {e}")
