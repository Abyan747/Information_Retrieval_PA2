"""
evaluate_query_expansion_with_VSM.py
Evaluates Pseudo-Relevance Feedback (Query Expansion) applied to the optimal
Pure Sparse Vector Space Model (TF_IDF with optimal c=0.85).

Explores:
  - Bo1QueryExpansion (Bose-Einstein 1 Divergence) across:
      * fb_docs in [3, 5, 10]
      * fb_terms in [5, 10, 15, 20, 25]
  - KLQueryExpansion (Kullback-Leibler Divergence) across:
      * fb_docs in [3, 5, 10]
      * fb_terms in [5, 10, 15, 20]

Measures:
  - MAP, nDCG@10, P@5, P@10, Recall@100
  - Paired statistical significance against unexpanded TF_IDF baseline

Outputs:
  - eval_query_expansion_vsm_results.txt : Detailed evaluation metrics & p-values
  - results_query_expansion_vsm.txt      : TREC run of the best query-expanded VSM model

Usage:
    python evaluate_query_expansion_with_VSM.py
"""

import os
import pandas as pd

import config
from config import (
    REL_FILE, PROCESSED_QRY_FILE, INDEX_DIR,
    GROUP_PREFIX
)
from info_fetch_search import parse_processed_queries, save_trec_run
from info_fetch_evaluate import parse_qrels

import pyterrier as pt
if not pt.java.started():
    pt.java.init()
import ir_measures
from ir_measures import MAP


def make_vsm_retriever(index, c_val: str = "0.85", num_results: int = 1000):
    """Create the optimal single-pass Sparse Vector Space Model (TF_IDF with c=0.85)."""
    return pt.terrier.Retriever(
        index,
        wmodel="TF_IDF",
        tokeniser="whitespace",
        num_results=num_results,
        controls={"c": str(c_val)},
    )


def main():
    print("=" * 65)
    print("Query Expansion on Optimal Pure Vector Space Model (VSM)")
    print("=" * 65)

    print(f"\nLoading index from '{INDEX_DIR}' ...")
    index = pt.IndexFactory.of(os.path.abspath(INDEX_DIR))

    print(f"Loading queries from '{PROCESSED_QRY_FILE}' ...")
    queries_df = parse_processed_queries(PROCESSED_QRY_FILE)
    print(f"  {len(queries_df)} queries loaded.")

    print(f"Loading relevance judgments from '{REL_FILE}' ...")
    qrels_df = parse_qrels(REL_FILE)
    print(f"  {len(qrels_df)} relevance judgments across {qrels_df['qid'].nunique()} queries.")

    # c=0.85 is the empirical peak found in the VSM sweep; c=0.75 is standard baseline
    base_opt = make_vsm_retriever(index, c_val="0.85")
    base_std = make_vsm_retriever(index, c_val="0.75")

    pipelines = []
    names = []

    names.append("TF_IDF_c0.75_unexpanded")
    pipelines.append(base_std)

    names.append("TF_IDF_c0.85_unexpanded")
    pipelines.append(base_opt)

    for fb_docs in [3, 5, 10]:
        for fb_terms in [5, 10, 15, 20, 25]:
            qe = pt.rewrite.Bo1QueryExpansion(index, fb_docs=fb_docs, fb_terms=fb_terms)
            names.append(f"VSM_Bo1_d{fb_docs}_t{fb_terms}")
            pipelines.append(base_opt >> qe >> base_opt)

    for fb_docs in [3, 5, 10]:
        for fb_terms in [5, 10, 15, 20]:
            qe = pt.rewrite.KLQueryExpansion(index, fb_docs=fb_docs, fb_terms=fb_terms)
            names.append(f"VSM_KL_d{fb_docs}_t{fb_terms}")
            pipelines.append(base_opt >> qe >> base_opt)

    baseline_idx = names.index("TF_IDF_c0.85_unexpanded")
    print(f"\nRunning experiment on {len(pipelines)} pipelines ...")
    print(f"Significance test baseline: '{names[baseline_idx]}'")

    eval_metrics = [MAP, ir_measures.nDCG@10, ir_measures.P@5, ir_measures.P@10, ir_measures.R@100]

    results = pt.Experiment(
        pipelines,
        queries_df,
        qrels_df,
        eval_metrics=eval_metrics,
        names=names,
        baseline=baseline_idx,
        filter_by_qrels=True,
    )

    map_col = "MAP" if "MAP" in results.columns else "AP"

    print(f"\n{'=' * 65}")
    print("QUERY EXPANSION (VSM) EVALUATION RESULTS")
    print(f"{'=' * 65}")
    summary_cols = ["name", map_col, "nDCG@10", "P@5", "P@10", "R@100"]
    existing_cols = [c for c in summary_cols if c in results.columns]
    print(results[existing_cols].to_string(index=False))

    out_table = "eval_query_expansion_vsm_results.txt"
    results.to_csv(out_table, index=False)
    print(f"\nFull evaluation metrics & p-values saved to '{out_table}'.")

    best_row = results.sort_values(by=map_col, ascending=False).iloc[0]
    best_name = best_row["name"]
    best_map = best_row[map_col]
    base_map = results.loc[results["name"] == "TF_IDF_c0.85_unexpanded", map_col].values[0]
    gain = ((best_map - base_map) / base_map) * 100

    print(f"\n{'=' * 65}")
    print(f"Optimal Expanded VSM Model: {best_name}")
    print(f"  Unexpanded MAP: {base_map:.4f}")
    print(f"  Expanded   MAP: {best_map:.4f}  (+{gain:.2f}%)")
    print(f"  nDCG@10:        {best_row['nDCG@10']:.4f}")
    print(f"  Recall@100:     {best_row['R@100']:.4f}")
    print(f"{'=' * 65}")

    best_pipe = pipelines[names.index(best_name)]
    print(f"\nGenerating TREC run for '{best_name}' ...")
    best_run = best_pipe.transform(queries_df)
    trec_out = "results_query_expansion_vsm.txt"
    save_trec_run(best_run, trec_out, run_tag="info_fetch_vsm_qe")
    print(f"TREC run saved to '{trec_out}'.")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
