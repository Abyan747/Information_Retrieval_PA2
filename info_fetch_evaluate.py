"""
info_fetch_evaluate.py
Evaluates all retrieval model runs against Cranfield relevance judgments.
Uses pt.Experiment() to compute MAP, NDCG@10, P@5, P@10, Recall@100.
Identifies the best model and overwrites info_fetch_results.txt with it.

Usage:
    python info_fetch_evaluate.py
"""

import os
import pandas as pd
import pyterrier as pt
import ir_measures
from ir_measures import MAP

from config import (
    REL_FILE, PROCESSED_QRY_FILE, INDEX_DIR,
    RESULTS_FILE, GROUP_PREFIX
)
from info_fetch_search import parse_processed_queries, save_trec_run


def parse_qrels(filepath: str) -> pd.DataFrame:
    """
    Parse Cranfield relevance judgments (cranqrel).
    Format: qid docno rel (space-separated, rel in 1-4)
    Returns DataFrame with columns: qid (str), docno (str), label (int)
    Relevance: 1-4 all treated as relevant (>=1). Negatives excluded.
    """
    rows = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                qid = str(int(parts[0]))
                docno = str(int(parts[1]))
                rel = int(parts[2])
                if rel >= 1:   # all 1-4 are relevant
                    rows.append({"qid": qid, "docno": docno, "label": rel})
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Info fetch -- PA2 Evaluation")
    print("=" * 60)

    # Load index
    print(f"\nLoading index from '{INDEX_DIR}' ...")
    index = pt.IndexFactory.of(os.path.abspath(INDEX_DIR))

    # Load queries and qrels
    print(f"Loading queries from '{PROCESSED_QRY_FILE}' ...")
    queries_df = parse_processed_queries(PROCESSED_QRY_FILE)

    print(f"Loading relevance judgments from '{REL_FILE}' ...")
    qrels_df = parse_qrels(REL_FILE)
    print(f"  {len(qrels_df)} relevance judgments, "
          f"{qrels_df['qid'].nunique()} queries with judgments")

    # Keep only queries that have relevance judgments
    valid_qids = set(qrels_df["qid"].unique())
    queries_df = queries_df[queries_df["qid"].isin(valid_qids)].reset_index(drop=True)
    print(f"  Using {len(queries_df)} queries that have relevance judgments")

    # ----------------------------------------------------------------
    # Define all pipelines
    # ----------------------------------------------------------------
    def make_retriever(wmodel, controls=None):
        return pt.BatchRetrieve(
            index,
            wmodel=wmodel,
            num_results=1000,
            controls=controls or {},
        )

    pipelines = [
        ("TF_IDF",       make_retriever("TF_IDF")),
        ("BM25_b0.1",    make_retriever("BM25", {"bm25.b": 0.1,  "bm25.k_1": 1.2})),
        ("BM25_b0.3",    make_retriever("BM25", {"bm25.b": 0.3,  "bm25.k_1": 1.2})),
        ("BM25_b0.5",    make_retriever("BM25", {"bm25.b": 0.5,  "bm25.k_1": 1.2})),
        ("BM25_b0.75",   make_retriever("BM25", {"bm25.b": 0.75, "bm25.k_1": 1.2})),
        ("BM25_b0.9",    make_retriever("BM25", {"bm25.b": 0.9,  "bm25.k_1": 1.2})),
        ("PL2_c0.1",     make_retriever("PL2",  {"c": 0.1})),
        ("PL2_c0.3",     make_retriever("PL2",  {"c": 0.3})),
        ("PL2_c0.5",     make_retriever("PL2",  {"c": 0.5})),
        ("PL2_c0.75",    make_retriever("PL2",  {"c": 0.75})),
        ("PL2_c1.0",     make_retriever("PL2",  {"c": 1.0})),
        ("In_expB2",     make_retriever("In_expB2")),
    ]

    names      = [p[0] for p in pipelines]
    retrievers = [p[1] for p in pipelines]

    # ----------------------------------------------------------------
    # Run pt.Experiment
    # ----------------------------------------------------------------
    print(f"\nRunning pt.Experiment() with {len(pipelines)} models ...")

    eval_metrics = [MAP, ir_measures.nDCG@10, ir_measures.P@5, ir_measures.P@10, ir_measures.R@100]

    results_table = pt.Experiment(
        retrievers,
        queries_df,
        qrels_df,
        eval_metrics=eval_metrics,
        names=names,
        baseline=0,          # compare everything to TF_IDF
    )

    # ----------------------------------------------------------------
    # Display and save results table
    # ----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(results_table.to_string(index=False))

    table_file = f"{GROUP_PREFIX}_eval_results.txt"
    results_table.to_csv(table_file, index=False)
    print(f"\nFull results table saved to '{table_file}'")

    # ----------------------------------------------------------------
    # Pick best model by MAP
    # ----------------------------------------------------------------
    best_row = results_table.loc[results_table["AP"].idxmax()]
    best_name = best_row["name"]
    best_map  = best_row["AP"]
    print(f"\nBest model: {best_name}  (MAP = {best_map:.4f})")

    # Generate submission run for best model on all 225 queries
    print(f"\nRe-running best model '{best_name}' on all {len(queries_df)} queries for submission ...")
    best_idx = names.index(best_name)
    best_retriever = retrievers[best_idx]
    submission_results = best_retriever.transform(queries_df)
    save_trec_run(submission_results, RESULTS_FILE, run_tag="info_fetch")
    print(f"Best model results saved to '{RESULTS_FILE}' (submission file).")

    print(f"\n{'=' * 60}")
    print("Summary for Report")
    print(f"{'=' * 60}")
    summary_cols = ["name", "AP", "nDCG@10", "P@5", "P@10", "R@100"]
    existing_cols = [c for c in summary_cols if c in results_table.columns]
    print(results_table[existing_cols].to_string(index=False))


if __name__ == "__main__":
    main()
