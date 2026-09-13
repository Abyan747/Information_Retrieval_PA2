"""
info_fetch_utils.py
Shared preprocessing utilities. Both the indexing and search programs
import from here so queries and documents go through an identical pipeline.

The four required preprocessing steps, in order:
    1. tokenize          - split text into alphabetic tokens
    2. normalize         - lowercase
    3. remove_stopwords  - drop common words
    4. stem_tokens       - Porter stemmer (NLTK)
"""

import re
from nltk.stem import PorterStemmer

# Module-level instance so we don't reconstruct the stemmer on every call.
_stemmer = PorterStemmer()


def load_stopwords(filepath: str) -> set:
    with open(filepath, encoding="utf-8", errors="replace") as f:
        return {line.strip().lower() for line in f if line.strip()}


def tokenize(text: str) -> list:
    """Step 1 - extract alphabetic runs; discard digits and punctuation."""
    return re.findall(r"[a-zA-Z]+", text)


def normalize(tokens: list) -> list:
    """Step 2 - lowercase so 'Aerodynamic' and 'aerodynamic' map to the same term."""
    return [t.lower() for t in tokens]


def remove_stopwords(tokens: list, stopwords: set) -> list:
    """Step 3 - drop high-frequency function words that carry no retrieval value."""
    return [t for t in tokens if t not in stopwords]


def stem_tokens(tokens: list) -> list:
    """Reduce inflected forms to a common root via Porter stemmer."""
    return [_stemmer.stem(t) for t in tokens]


def preprocess(text: str, stopwords: set) -> list:
    """Run the full four-step pipeline on a raw text string."""
    tokens = tokenize(text)
    tokens = normalize(tokens)
    tokens = remove_stopwords(tokens, stopwords)
    tokens = stem_tokens(tokens)
    return tokens


def preprocess_term(word: str, stopwords: set) -> str:
    """
    Preprocess a single query word.
    Returns "" if the word is a stop word or contains no alphabetic characters,
    so the caller can detect and handle the case without an extra lookup.
    """
    result = preprocess(word, stopwords)
    return result[0] if result else ""
