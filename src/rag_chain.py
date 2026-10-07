import re

from gemini_llm import clean_response, get_llm
from prompt_template import get_medical_prompt
from retriever import retrieve_documents

TELUGU_SCRIPT = re.compile(r"[\u0C00-\u0C7F]")

NOT_FOUND_MESSAGE = (
    "I could not find this information in the provided medical documents."
)


def translate_for_search(llm, question):
    """The PDFs are in English. If the question is in Telugu script,
    translate it to English so keyword search and query expansion work.
    The original question is still used for the final answer."""

    if not TELUGU_SCRIPT.search(question):
        return question

    try:
        response = llm.invoke(
            "Translate this medical question into a short English search "
            "query. Return ONLY the English text.\n\n" + question
        )
        return clean_response(response) or question

    except Exception as e:
        print("Translation failed, using original question:", e)
        return question


def build_context(documents):

    parts = []

    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")

        parts.append(f"[Source: {source}, page {page}]\n{doc.page_content}")

    return "\n\n".join(parts)


def create_rag_chain():

    prompt = get_medical_prompt()
    llm = get_llm()

    def ask_question(question):

        question = question.strip()

        # 1. English search query (only differs for Telugu script)
        search_question = translate_for_search(llm, question)

        # 2. Retrieve the best chunks
        documents = retrieve_documents(question, search_question)

        if not documents:
            return NOT_FOUND_MESSAGE

        # 3. Build the prompt
        formatted_prompt = prompt.invoke(
            {
                "context": build_context(documents),
                "question": question,
            }
        )

        # 4. Ask Gemini and clean the response
        response = llm.invoke(formatted_prompt)

        return clean_response(response)

    return ask_question


# ==========================================
# TEST RAG
# ==========================================

if __name__ == "__main__":

    rag_chain = create_rag_chain()

    answer = rag_chain("What is hypertension?")

    print("\n==============================")
    print("MEDICAL RAG ANSWER")
    print("==============================")

    print(answer)