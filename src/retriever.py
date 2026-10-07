import hashlib
import os
import re
from functools import lru_cache

from langchain_chroma import Chroma

from config import (
    FINAL_TOP_K,
    KEYWORD_CAP,
    KEYWORD_WEIGHT,
    TOP_K_PER_QUERY,
    VECTORSTORE_PATH,
)
from embeddings import get_embedding_model
from medicine_mapping import find_generic_names

DEBUG = os.getenv("RAG_DEBUG", "1") == "1"

REFERENCE_PENALTY = 0.25   # added to the distance of bibliography-like chunks


# =========================================================
# VECTOR STORE (loaded once, reused for every question)
# =========================================================

@lru_cache(maxsize=1)
def get_vector_store():

    if not VECTORSTORE_PATH.exists():
        raise FileNotFoundError(
            f"Vector store not found at {VECTORSTORE_PATH}. "
            "Run: python src/vector_store.py"
        )

    return Chroma(
        persist_directory=str(VECTORSTORE_PATH),
        embedding_function=get_embedding_model(),
    )


# =========================================================
# SMALL HELPERS
# =========================================================

def has_word(text, phrase):
    """Whole word / phrase match."""
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def has_stem(text, stem):
    """Word-start match, so 'symptom' also matches 'symptoms'."""
    return re.search(rf"\b{re.escape(stem)}", text) is not None


DEFINITION_PHRASES = [
    "what is", "what are", "meaning of", "define",
    "definition", "ante enti", "enti", "ante emiti",
]

SYMPTOM_WORDS = ["symptom", "sign", "lakshana"]
CAUSE_WORDS = ["cause", "risk factor", "karana"]
TREATMENT_WORDS = ["treatment", "management", "therapy", "chikitsa"]

# Filler words removed when we have to guess the topic
STOP_WORDS = {
    "what", "is", "are", "the", "of", "for", "a", "an", "how", "does",
    "do", "to", "in", "about", "tell", "me", "explain",
    "ante", "enti", "emiti", "deniki", "use", "uses", "chestaru",
    "ela", "em", "ki", "gurinchi", "tablet", "tablets", "medicine",
    "symptoms", "symptom", "causes", "cause", "treatment",    "give", "first", "aid", "steps", "step", "treat", "can", "should",
    "i", "my", "you",
}


def any_phrase(text, phrases):
    return any(has_word(text, p) for p in phrases)


def any_stem(text, stems):
    return any(has_stem(text, s) for s in stems)


def looks_like_references(text):
    """True for bibliography / citation chunks (they match topic words
    but contain no useful information)."""

    t = text.lower()

    markers = ["doi.org", "doi:", "et al", "accessed", "http", "pubmed"]

    hits = sum(t.count(m) for m in markers)

    return hits >= 2


# =========================================================
# EXTRACT MAIN TOPIC FROM QUESTION
# =========================================================

def extract_topic(question):

    q = question.lower().strip().rstrip("?.! ")

    patterns = [
        # English
        r"^(?:what is|what are)\s+(?:the\s+)?(.+)$",
        r"^(?:symptoms|signs|causes|treatment|management|meaning)\s+of\s+(.+)$",
        r"^define\s+(.+)$",
        # Telugu written in English letters
        r"^(.+?)\s+(?:ante|anna)\b.*$",
        r"^(.+?)\s+(?:gurinchi|gurunchi)\b.*$",
        r"^(.+?)\s+(?:yokka\s+)?(?:lakshana|karana|chikitsa)\w*.*$",
    ]

    topic = None

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            topic = match.group(1).strip()
            break

    if topic is None:
        # No pattern matched: drop filler words and keep the rest
        words = [w for w in re.findall(r"[\w'-]+", q) if w not in STOP_WORDS]
        topic = " ".join(words)

    topic = re.sub(r"\b(disease|condition|problem)\b", "", topic)

    return re.sub(r"\s+", " ", topic).strip() or q



# =========================================================
# BUILD SEARCH QUERIES
# =========================================================

def build_search_queries(question):

    q = question.lower().strip()
    topic = extract_topic(question)

    queries = [question]

    # Definition / information questions
    if any_phrase(q, DEFINITION_PHRASES):
        queries += [
            f"{topic} definition",
            f"what is {topic}",
            f"{topic} overview",
            f"{topic} symptoms causes",
        ]

    if any_stem(q, SYMPTOM_WORDS):
        queries += [
            f"{topic} symptoms",
            f"signs and symptoms of {topic}",
        ]

    if any_stem(q, CAUSE_WORDS):
        queries += [
            f"{topic} causes",
            f"risk factors for {topic}",
        ]

    if any_stem(q, TREATMENT_WORDS):
        queries += [
            f"{topic} treatment",
            f"{topic} management",
        ]
        # How-to / first-aid questions
    if has_word(q, "first aid") or has_word(q, "how to"):
        queries += [
            f"{topic} first aid",
            f"first aid for {topic}",
            f"{topic} treatment steps",
            f"how to treat {topic}",
        ]
    # Medicine (brand -> generic)
    for generic in find_generic_names(question):
        queries += [
            generic,
            f"{generic} uses",
            f"{generic} indication",
            f"{generic} medicine information",
        ]

    # Remove duplicates, keep order
    unique_queries = []

    for query in queries:
        query = query.strip()
        if query and query not in unique_queries:
            unique_queries.append(query)

    return unique_queries


# =========================================================
# KEYWORD SCORING
# =========================================================

def calculate_keyword_score(question, document):

    q = question.lower()
    text = document.page_content.lower()
    topic = extract_topic(question)

    score = 0

    # Topic words found in the chunk
    for word in set(topic.split()):
        if len(word) > 3 and has_stem(text, word):
            score += 6

    # Definition question
    if any_phrase(q, DEFINITION_PHRASES):
        for kw in ["defined as", "refers to", "is a disease",
                   "is a condition", "disorder", "definition"]:
            if kw in text:
                score += 5

    # Symptom question
    if any_stem(q, SYMPTOM_WORDS):
        for kw in ["symptom", "signs", "clinical features"]:
            if has_stem(text, kw):
                score += 6

    # Cause question
    if any_stem(q, CAUSE_WORDS):
        for kw in ["cause", "risk factor"]:
            if has_stem(text, kw):
                score += 6

    # Treatment question
    if any_stem(q, TREATMENT_WORDS):
        for kw in ["treatment", "management", "therapy", "prevention"]:
            if has_stem(text, kw):
                score += 6

    # Medicine question
    generics = find_generic_names(question)

    if generics:
        for generic in generics:
            if has_stem(text, generic):
                score += 10

        if has_stem(text, "uses") or has_stem(text, "indication"):
            score += 5

    return score


# =========================================================
# RETRIEVE DOCUMENTS
# =========================================================

def retrieve_documents(question, search_question=None):
    """
    question        : the user's original question
    search_question : optional English version used only for searching
                      (e.g. translated from Telugu script)
    """

    query_text = search_question or question

    vector_store = get_vector_store()

    search_queries = build_search_queries(query_text)

    if DEBUG:
        print("\n========== USER QUERY ==========")
        print(question)
        if search_question and search_question != question:
            print("Search version:", search_question)
        print("\n========== SEARCH QUERIES ==========")
        for query in search_queries:
            print(" -", query)

    # ---------------- vector search ----------------
    all_results = []

    for query in search_queries:
        all_results.extend(
            vector_store.similarity_search_with_score(
                query, k=TOP_K_PER_QUERY
            )
        )

    # ---------------- de-duplicate -----------------
    best = {}

    for doc, vector_score in all_results:

        # one entry per unique chunk (not per page)
        key = hashlib.md5(
            doc.page_content.encode("utf-8")
        ).hexdigest()

        keyword_score = calculate_keyword_score(query_text, doc)

        # Chroma returns a DISTANCE: smaller = better.
        # Keyword bonus is capped so it cannot swamp the vector score.
        final_score = vector_score - (
            min(keyword_score, KEYWORD_CAP) * KEYWORD_WEIGHT
        )

        # Bibliography / citation chunks are pushed down
        if looks_like_references(doc.page_content):
            final_score += REFERENCE_PENALTY

        if key not in best or final_score < best[key][3]:
            best[key] = (doc, vector_score, keyword_score, final_score)

    ranked = sorted(best.values(), key=lambda x: x[3])
    final_results = ranked[:FINAL_TOP_K]

    # ---------------- debug print ------------------
    if DEBUG:
        print("\n========== RETRIEVED DOCUMENTS ==========")
        for i, (doc, vs, ks, fs) in enumerate(final_results, 1):
            print(
                f"\n--- Document {i} --- "
                f"{doc.metadata.get('source', '?')} "
                f"p.{doc.metadata.get('page', '?')} | "
                f"vector={vs:.3f} keyword={ks} final={fs:.3f}"
            )
            print(doc.page_content[:300].replace("\n", " "))
        print(f"\nTotal documents retrieved: {len(final_results)}")

    return [doc for doc, _, _, _ in final_results]


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    retrieve_documents("What is diabetes?")