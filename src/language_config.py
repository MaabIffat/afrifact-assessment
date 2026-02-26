SUPPORTED_LANGUAGES = {
    "amharic": "Amharic",
    "english": "English",
    "hausa": "Hausa",
    "swahili": "Swahili",
    "yoruba": "Yoruba",
    "zulu": "Zulu",
}


def get_language_instruction(language_key):
    if language_key not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Language not supported: {language_key}")

    return f"""
The claim and evidence are written in {SUPPORTED_LANGUAGES[language_key]}.
Perform fact-checking in the same language.
Do NOT translate unless absolutely necessary for reasoning.
Provide final justification in {SUPPORTED_LANGUAGES[language_key]}.
"""
