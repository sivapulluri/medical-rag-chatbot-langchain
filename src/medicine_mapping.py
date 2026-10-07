# src/medicine_mapping.py

import re


# Brand name -> Generic medicine name
MEDICINE_MAPPING = {

    # Paracetamol
    "dolo 650": "paracetamol",
    "dolo": "paracetamol",
    "crocin": "paracetamol",
    "calpol": "paracetamol",

    # Pantoprazole
    "pantocid": "pantoprazole",
    "pan 40": "pantoprazole",
    "pantop": "pantoprazole",

    # Cetirizine
    "cetzine": "cetirizine",
    "alerid": "cetirizine",

    # Telmisartan
    "telma": "telmisartan",
    "telsar": "telmisartan",
}


def _contains_word(text, phrase):
    """Whole-word / whole-phrase match (so 'pan 40' != 'span 400')."""
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def find_generic_names(question):
    """Return every generic medicine name referred to in the question
    (by brand name), without duplicates, in order of appearance."""

    q = question.lower()
    generics = []

    for brand_name, generic_name in MEDICINE_MAPPING.items():
        if _contains_word(q, brand_name) and generic_name not in generics:
            generics.append(generic_name)

    return generics


def normalize_medicine_query(question):
    """
    Add the generic medicine name(s) to a brand-name medicine query.

    Example:
        Dolo 650 deniki use chestaru?
        ->
        Dolo 650 deniki use chestaru? paracetamol
    """

    q = question.lower()

    missing = [
        g for g in find_generic_names(question)
        if not _contains_word(q, g)
    ]

    if not missing:
        return question

    return f"{question} {' '.join(missing)}"


if __name__ == "__main__":

    test_questions = [
        "Dolo 650 deniki use chestaru?",
        "Crocin deniki use chestaru?",
        "Calpol uses enti?",
        "Pantocid deniki?",
        "Cetzine tablet use enti?",
        "Telma tablet deniki use chestaru?",
        "Paracetamol deniki use chestaru?",
        "Dolo and Pantocid difference enti?",
    ]

    print("\nMedicine Mapping Test")
    print("====================")

    for question in test_questions:

        result = normalize_medicine_query(question)

        print("\nOriginal :", question)
        print("Search   :", result)