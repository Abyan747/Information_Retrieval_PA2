"""
info_fetch_index.py
Builds a PyTerrier inverted index from the preprocessed Cranfield corpus
(info_fetch_processed.all). Terrier's own stemmer and stopper are disabled
because the text has already been preprocessed by our PA1 pipeline.

Records and prints:
  - Indexing time (seconds)
  - Index size on disk (KB)
  - Vocabulary size and number of postings

Usage:
    python info_fetch_index.py
"""

import os
import time
import shutil

# Set JAVA_HOME before any pyterrier import
from config import PROCESSED_FILE, INDEX_DIR, GROUP_PREFIX

import pyterrier as pt


def parse_processed_corpus(filepath: str):
    """
    Generator that yields {"docno": str, "text": str} dicts
    from info_fetch_processed.all.
    Output format of preprocess: two lines per doc:
        .I <docid>
        .S <stem1> <stem2> ...
    """
    docno = None
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith(".I "):
                docno = line[3:].strip()
            elif line.startswith(".S ") and docno is not None:
                text = line[3:].strip()
                yield {"docno": docno, "text": text}
                docno = None


def get_dir_size_kb(path: str) -> float:
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1024.0


def main():
    print("=" * 60)
    print("Info fetch -- PA2 Indexer")
    print("=" * 60)

    # Remove old index if exists
    if os.path.exists(INDEX_DIR):
        print(f"\nRemoving existing index at '{INDEX_DIR}' ...")
        shutil.rmtree(INDEX_DIR)

    print(f"\nLoading preprocessed corpus from '{PROCESSED_FILE}' ...")
    docs = list(parse_processed_corpus(PROCESSED_FILE))
    print(f"  {len(docs)} documents loaded.")

    print(f"\nBuilding index at '{INDEX_DIR}' ...")
    print("  (Terrier stemmer/stopper disabled -- PA1 pipeline already applied)")

    indexer = pt.IterDictIndexer(
        os.path.abspath(INDEX_DIR),
        overwrite=True,
        meta={"docno": 10, "text": 16384},
        stemmer=None,
        stopwords=None,
        tokeniser="UTFTokeniser",
    )

    t_start = time.time()
    index_ref = indexer.index(iter(docs))
    t_elapsed = time.time() - t_start

    index = pt.IndexFactory.of(index_ref)
    stats = index.getCollectionStatistics()

    index_size_kb = get_dir_size_kb(INDEX_DIR)

    print(f"\n{'=' * 60}")
    print(f"  Indexing complete!")
    print(f"{'=' * 60}")
    print(f"  Documents indexed : {stats.getNumberOfDocuments():,}")
    print(f"  Unique terms      : {stats.getNumberOfUniqueTerms():,}")
    print(f"  Total postings    : {stats.getNumberOfPointers():,}")
    print(f"  Total tokens      : {stats.getNumberOfTokens():,}")
    print(f"  Indexing time     : {t_elapsed:.3f} seconds")
    print(f"  Index size        : {index_size_kb:.1f} KB  ({index_size_kb/1024:.2f} MB)")
    print(f"{'=' * 60}")

    stats_file = f"{GROUP_PREFIX}_index_stats.txt"
    with open(stats_file, "w") as f:
        f.write(f"Documents indexed : {stats.getNumberOfDocuments()}\n")
        f.write(f"Unique terms      : {stats.getNumberOfUniqueTerms()}\n")
        f.write(f"Total postings    : {stats.getNumberOfPointers()}\n")
        f.write(f"Total tokens      : {stats.getNumberOfTokens()}\n")
        f.write(f"Indexing time (s) : {t_elapsed:.3f}\n")
        f.write(f"Index size (KB)   : {index_size_kb:.1f}\n")
    print(f"\nStats saved to '{stats_file}'")


if __name__ == "__main__":
    main()
