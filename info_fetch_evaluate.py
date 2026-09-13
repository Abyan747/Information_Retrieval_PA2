"""
info_fetch_evaluate.py
Evaluates all Sparse Vector Space Model retrieval pipelines against Cranfield
relevance judgments (cranqrel) using modern PyTerrier.

Complies strictly with PA2 requirements:
    "You may use only sparse vector space models. Don't use advanced probabilistic
     or dense neural retrieval models."

Evaluates:
  - MAP, NDCG@10, P@5, P@10, Recall@100
  - Paired statistical significance tests (p-values) against baseline

Modern PyTerrier optimizations:
  - Uses pt.terrier.Retriever() (replaces deprecated pt.BatchRetrieve)
  - tokeniser="whitespace"
  - filter_by_qrels=True natively in pt.Experiment()
  - Dynamic MAP / AP column detection (avoids KeyError across ir_measures versions)
  - Vectorized pt.io.write_results() for submission file generation

Outputs:
  - info_fetch_eval_results.txt : Full evaluation table with metrics & p-values
  - info_fetch_results.txt      : TREC-format ranked results for the best model

Usage:
    python info_fetch_evaluate.py
"""

import os
import pandas as pd

from config import (
    REL_FILE, PROCESSED_QRY_FILE, INDEX_DIR,
    RESULTS_FILE, GROUP_PREFIX
)
from info_fetch_search import parse_processed_queries, save_trec_run

import pyterrier as pt
if not pt.java.started():
    pt.java.init()
import ir_measures
from ir_measures import MAP


def parse_qrels(filepath: str) -> pd.DataFrame:
    """
    Parse Cranfield relevance judgments (cranqrel).
    Format: qid docno rel (space-separated, rel in 1-4).
    Relevance: 1-4 all treated as relevant (>=1). Negative scores excluded.
    Returns DataFrame with columns: qid (str), docno (str), label (int).
    """
    rows = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                qid = str(int(parts[0]))
                docno = str(int(parts[1]))
                rel = int(parts[2])
                if rel >= 1:
                    rows.append({"qid": qid, "docno": docno, "label": rel})
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Info fetch -- PA2 Evaluation (Modern PyTerrier)")
    print("=" * 60)

    # Load index
    print(f"\nLoading index from '{INDEX_DIR}' ...")
    index = pt.IndexFactory.of(os.path.abspath(INDEX_DIR))

    # Load queries and qrels
    print(f"Loading queries from '{PROCESSED_QRY_FILE}' ...")
    queries_df = parse_processed_queries(PROCESSED_QRY_FILE)
    print(f"  {len(queries_df)} queries loaded.")

    print(f"Loading relevance judgments from '{REL_FILE}' ...")
    qrels_df = parse_qrels(REL_FILE)
    print(f"  {len(qrels_df)} relevance judgments, {qrels_df['qid'].nunique()} queries with judgments")

    # ----------------------------------------------------------------
    # Define All Sparse Vector Space Model Pipelines
    # ----------------------------------------------------------------
    def make_vsm_retriever(wmodel: str, controls: dict = None):
        return pt.terrier.Retriever(
            index,
            wmodel=wmodel,
            tokeniser="whitespace",
            num_results=1000,
            controls=controls or {},
        )

    pipelines = []
    names = []

    # 1. TF_IDF document length normalization parameter sweep (c)
    for c_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.5, 2.0]:
        names.append(f"TF_IDF_c{c_val}")
        pipelines.append(make_vsm_retriever("TF_IDF", {"c": str(c_val)}))

    # 2. Raw Term Frequency baseline
    names.append("Tf")
    pipelines.append(make_vsm_retriever("Tf"))

    # 3. Coordinate Match baseline
    names.append("CoordinateMatch")
    pipelines.append(make_vsm_retriever("CoordinateMatch"))

    # 4. Lemur TF-IDF Vector Space formulation
    names.append("LemurTF_IDF")
    pipelines.append(make_vsm_retriever("LemurTF_IDF"))

    # ----------------------------------------------------------------
    # Run pt.Experiment with filter_by_qrels=True
    # Baseline set to TF_IDF_c0.75 (standard default) for significance testing
    # ----------------------------------------------------------------
    baseline_idx = names.index("TF_IDF_c0.75")
    print(f"\nRunning pt.Experiment() with {len(pipelines)} pure Sparse VSM pipelines ...")
    print(f"  Baseline for significance tests: '{names[baseline_idx]}'")

    eval_metrics = [MAP, ir_measures.nDCG@10, ir_measures.P@5, ir_measures.P@10, ir_measures.R@100]

    results_table = pt.Experiment(
        pipelines,
        queries_df,
        qrels_df,
        eval_metrics=eval_metrics,
        names=names,
        baseline=baseline_idx,
        filter_by_qrels=True,
    )

    # ----------------------------------------------------------------
    # Display and Save Results Table
    # ----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(results_table.to_string(index=False))

    table_file = f"{GROUP_PREFIX}_eval_results.txt"
    results_table.to_csv(table_file, index=False)
    print(f"\nFull results table saved to '{table_file}'")

    # ----------------------------------------------------------------
    # Determine Best Model Dynamically (handle "MAP" vs "AP" column name)
    # ----------------------------------------------------------------
    map_col = "MAP" if "MAP" in results_table.columns else "AP"
    best_row = results_table.loc[results_table[map_col].idxmax()]
    best_name = best_row["name"]
    best_map = best_row[map_col]
    print(f"\nBest Sparse VSM Model: {best_name}  (MAP = {best_map:.4f})")

    # Re-run best model on all queries to generate official submission run
    print(f"\nGenerating submission results for best model '{best_name}' on all {len(queries_df)} queries ...")
    best_retriever = pipelines[names.index(best_name)]
    submission_results = best_retriever.transform(queries_df)
    save_trec_run(submission_results, RESULTS_FILE, run_tag="info_fetch")
    print(f"Best model results saved to '{RESULTS_FILE}' (submission file).")

    # ----------------------------------------------------------------
    # Summary for Report
    # ----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Summary for Report")
    print(f"{'=' * 60}")
    summary_cols = ["name", map_col, "nDCG@10", "P@5", "P@10", "R@100"]
    existing_cols = [c for c in summary_cols if c in results_table.columns]
    print(results_table[existing_cols].to_string(index=False))


if __name__ == "__main__":
    main()
