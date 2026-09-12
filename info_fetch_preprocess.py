"""
info_fetch_preprocess.py
Reads cran.all.1400, applies the four-step preprocessing pipeline to each
document's Title (.T) and Abstract (.W), and writes the result to
info_fetch_processed.all.

Author (.A) and Bibliography (.B) sections are intentionally skipped -
they contain names and citation metadata that would pollute the index.

Output format (two lines per document):
    .I <docid>
    .S <stem1> <stem2> ...

Usage:
    python info_fetch_preprocess.py
"""

import os
import sys
from config import CORPUS_FILE, STOPWORDS_FILE, PROCESSED_FILE, GROUP_PREFIX
from info_fetch_utils import load_stopwords, preprocess


def parse_cranfield(filepath: str) -> list:
    """
    Line-by-line state machine parser for the SGML-like Cranfield format.
    Returns a list of dicts with keys: id, title, body.
    """
    documents = []
    current_doc = None
    section = None

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")

            if line.startswith(".I "):
                if current_doc is not None:
                    documents.append(current_doc)
                current_doc = {"id": int(line[3:].strip()), "title": "", "body": ""}
                section = None
            elif line == ".T":
                section = "title"
            elif line in (".A", ".B"):
                section = None
            elif line == ".W":
                section = "body"
            elif current_doc is not None:
                if section == "title":
                    current_doc["title"] += " " + line
                elif section == "body":
                    current_doc["body"] += " " + line

    if current_doc is not None:
        documents.append(current_doc)

    return documents


def parse_cranfield_queries(filepath: str) -> list:
    """
    Parse Cranfield query file (.I and .W sections).
    Returns a list of dicts with keys: id, query.
    """
    queries = []
    current_q = None
    in_w = False

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith(".I "):
                if current_q is not None:
                    current_q["query"] = current_q["query"].strip()
                    queries.append(current_q)
                # Cranfield relevance judgments (cranqrel) index queries 1..225
                # corresponding to sequential position in cran.qry.
                current_q = {"id": len(queries) + 1, "query": ""}
                in_w = False
            elif line == ".W":
                in_w = True
            elif in_w and current_q is not None:
                current_q["query"] += " " + line

    if current_q is not None:
        current_q["query"] = current_q["query"].strip()
        queries.append(current_q)

    return queries


import argparse

def main():
    parser = argparse.ArgumentParser(description="Preprocess the Cranfield collection.")
    parser.add_argument("corpus", nargs="?", default=CORPUS_FILE, help="Path to input corpus")
    parser.add_argument("stopwords", nargs="?", default=STOPWORDS_FILE, help="Path to stopwords file")
    parser.add_argument("output", nargs="?", default=PROCESSED_FILE, help="Path to output processed file")
    parser.add_argument("--query-file", default=None, help="Path to query file (e.g., cran.qry)")
    parser.add_argument("--query-out", default=None, help="Path to output processed query file")
    args = parser.parse_args()

    print(f"Loading stop words from '{args.stopwords}' ...")
    try:
        stopwords = load_stopwords(args.stopwords)
    except FileNotFoundError:
        print(f"  Error: '{args.stopwords}' not found.")
        sys.exit(1)
    print(f"  {len(stopwords)} stop words loaded.")

    print(f"\nParsing corpus from '{args.corpus}' ...")
    try:
        documents = parse_cranfield(args.corpus)
    except FileNotFoundError:
        print(f"  Error: '{args.corpus}' not found.")
        sys.exit(1)
    print(f"  {len(documents)} documents found.")

    print("\nApplying preprocessing pipeline ...")
    print("  tokenize -> normalize -> remove stop words -> stem")

    with open(args.output, "w", encoding="utf-8") as out:
        for doc in documents:
            tokens = preprocess(doc["title"] + " " + doc["body"], stopwords)
            out.write(f".I {doc['id']}\n")
            out.write(f".S {' '.join(tokens)}\n")

    print(f"\nCorpus output written to '{args.output}' ({len(documents)} documents).")

    # Optionally preprocess queries if provided or if cran.qry exists
    query_file = args.query_file or ("cran.qry" if os.path.exists("cran.qry") else None)
    if query_file and os.path.exists(query_file):
        query_out = args.query_out or f"{GROUP_PREFIX}_processed_queries.txt"
        print(f"\nParsing queries from '{query_file}' ...")
        queries = parse_cranfield_queries(query_file)
        print(f"  {len(queries)} queries found.")
        with open(query_out, "w", encoding="utf-8") as out:
            for q in queries:
                tokens = preprocess(q["query"], stopwords)
                out.write(f".I {q['id']}\n")
                out.write(f".W {' '.join(tokens)}\n")
        print(f"Queries output written to '{query_out}' ({len(queries)} queries).")


if __name__ == "__main__":
    main()

