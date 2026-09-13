"""
info_fetch_evaluate_extended.py
Extended evaluation for PA2 report covering advanced lexical models:

  SECTION 1 (Report Part 2) - BM25 & DFR Model Tuning + Optimal Hybridization
    - BM25 fine-grained parameter tuning:
        * Length normalization sweep: b in [0.3, 0.5, 0.6, 0.7, 0.72, 0.75, 0.78, 0.8, 0.85, 0.9, 1.0]
        * Frequency saturation sweep: k_1 in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0] at optimal b
    - In_expB2 fine-grained parameter tuning:
        * Normalization parameter sweep: c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]
    - Additional DFR baselines: PL2, BB2, DPH, DFRee
    - Best-of-Both Hybridization:
        * Linear score interpolation combining the individually optimal BM25 and In_expB2:
          score = alpha * BM25* + (1 - alpha) * In_expB2*  for alpha in [0.1 .. 0.9]

  SECTION 2 (Report Part 3) - Query Expansion on Advanced Models
    - Bo1 and KL Pseudo-Relevance Feedback applied to:
        * Optimal BM25 retriever
        * Optimal In_expB2 retriever
        * Optimal BM25+In_expB2 Hybrid ensemble

NOTE: Pure Sparse Vector Space Models (TF_IDF sweeps, Tf, CoordinateMatch,
      LemurTF_IDF) reside exclusively in info_fetch_evaluate.py for submission compliance.

Outputs:
  - info_fetch_eval_results_extended.txt : Evaluation table with metrics & p-values (CSV)
  - info_fetch_results_extended.txt      : TREC-format run of overall best model

Usage:
    python info_fetch_evaluate_extended.py
"""

import os
import pandas as pd

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


# ============================================================
# Helpers
# ============================================================

def make_retriever(index, wmodel: str, controls: dict = None, num_results: int = 1000):
    """Build a pt.terrier.Retriever with whitespace tokenizer."""
    return pt.terrier.Retriever(
        index,
        wmodel=wmodel,
        tokeniser="whitespace",
        num_results=num_results,
        controls=controls or {},
    )


def run_experiment(pipelines, names, queries_df, qrels_df, baseline_name, section_label):
    """Run pt.Experiment for a section, print summary, and return results table."""
    baseline_idx = names.index(baseline_name) if baseline_name in names else 0
    print(f"\n{'=' * 65}")
    print(f"  {section_label}")
    print(f"  {len(pipelines)} pipelines  |  baseline = '{names[baseline_idx]}'")
    print(f"{'=' * 65}")

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
    summary_cols = ["name", map_col, "nDCG@10", "P@5", "P@10", "R@100"]
    existing = [c for c in summary_cols if c in results.columns]
    print(results[existing].to_string(index=False))
    return results, map_col


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print("Info fetch -- PA2 Extended Evaluation (Advanced Models)")
    print("=" * 65)

    # Load shared resources
    print(f"\nLoading index from '{INDEX_DIR}' ...")
    index = pt.IndexFactory.of(os.path.abspath(INDEX_DIR))

    print(f"Loading queries from '{PROCESSED_QRY_FILE}' ...")
    queries_df = parse_processed_queries(PROCESSED_QRY_FILE)
    print(f"  {len(queries_df)} queries loaded.")

    print(f"Loading relevance judgments from '{REL_FILE}' ...")
    qrels_df = parse_qrels(REL_FILE)
    print(f"  {len(qrels_df)} relevance judgments, {qrels_df['qid'].nunique()} queries with judgments")

    all_section_results = []

    p2_names, p2_pipes = [], []

    b_values = [0.3, 0.5, 0.6, 0.7, 0.72, 0.75, 0.78, 0.8, 0.85, 0.9, 1.0]
    for b in b_values:
        p2_names.append(f"BM25_b{b}")
        p2_pipes.append(make_retriever(index, "BM25", {"bm25.b": str(b)}))

    k1_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    for k1 in k1_values:
        if k1 == 1.2:
            continue  # Already evaluated above as BM25_b0.75
        p2_names.append(f"BM25_b0.75_k1_{k1}")
        p2_pipes.append(make_retriever(index, "BM25", {"bm25.b": "0.75", "bm25.k_1": str(k1)}))

    inexpb2_c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]
    for c in inexpb2_c_values:
        p2_names.append(f"In_expB2_c{c}")
        p2_pipes.append(make_retriever(index, "In_expB2", {"c": str(c)}))

    for c in [0.5, 1.0, 1.5]:
        p2_names.append(f"PL2_c{c}")
        p2_pipes.append(make_retriever(index, "PL2", {"c": str(c)}))
        p2_names.append(f"BB2_c{c}")
        p2_pipes.append(make_retriever(index, "BB2", {"c": str(c)}))

    for wmodel in ["DPH", "DFRee"]:
        p2_names.append(wmodel)
        p2_pipes.append(make_retriever(index, wmodel))

    # Individually optimal components identified from the parameter sweeps:
    #   - BM25 optimal: b=0.75, k_1=2.0 (MAP ~0.3217)
    #   - In_expB2 optimal: c=0.7 (MAP ~0.3312)
    #   - Standard BM25 baseline: b=0.75, k_1=1.2 (MAP ~0.3175)
    bm25_tuned = make_retriever(index, "BM25", {"bm25.b": "0.75", "bm25.k_1": "2.0"})
    bm25_std = make_retriever(index, "BM25", {"bm25.b": "0.75"})
    inexpb2_best = make_retriever(index, "In_expB2", {"c": "0.7"})

    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p2_names.append(f"Hybrid_TunedBM25+BestInExp_a{alpha}")
        p2_pipes.append(alpha * bm25_tuned + (1 - alpha) * inexpb2_best)

    for alpha in [0.1, 0.3, 0.5]:
        p2_names.append(f"Hybrid_StdBM25+BestInExp_a{alpha}")
        p2_pipes.append(alpha * bm25_std + (1 - alpha) * inexpb2_best)

    res2, mc2 = run_experiment(
        p2_pipes, p2_names, queries_df, qrels_df,
        baseline_name="BM25_b0.75",
        section_label="PART 2 -- BM25 & DFR Tuning + Best-of-Both Hybridization"
    )
    res2["section"] = "Part2_BM25_DFR_Hybrid"
    all_section_results.append(res2)

    bm25_qe_base = bm25_tuned
    inexpb2_qe_base = inexpb2_best

    p3_names, p3_pipes = [], []

    for fb_docs, fb_terms in [(3, 10), (3, 15), (3, 20), (5, 10), (5, 15), (5, 20), (10, 15)]:
        qe = pt.rewrite.Bo1QueryExpansion(index, fb_docs=fb_docs, fb_terms=fb_terms)
        p3_names.append(f"BM25_Bo1_d{fb_docs}_t{fb_terms}")
        p3_pipes.append(bm25_qe_base >> qe >> bm25_qe_base)

    for fb_docs, fb_terms in [(3, 10), (5, 15)]:
        qe = pt.rewrite.KLQueryExpansion(index, fb_docs=fb_docs, fb_terms=fb_terms)
        p3_names.append(f"BM25_KL_d{fb_docs}_t{fb_terms}")
        p3_pipes.append(bm25_qe_base >> qe >> bm25_qe_base)

    for fb_docs, fb_terms in [(3, 10), (3, 15), (5, 15), (5, 20)]:
        qe = pt.rewrite.Bo1QueryExpansion(index, fb_docs=fb_docs, fb_terms=fb_terms)
        p3_names.append(f"In_expB2_Bo1_d{fb_docs}_t{fb_terms}")
        p3_pipes.append(inexpb2_qe_base >> qe >> inexpb2_qe_base)

    for fb_docs, fb_terms in [(3, 10), (5, 15)]:
        qe = pt.rewrite.KLQueryExpansion(index, fb_docs=fb_docs, fb_terms=fb_terms)
        p3_names.append(f"In_expB2_KL_d{fb_docs}_t{fb_terms}")
        p3_pipes.append(inexpb2_qe_base >> qe >> inexpb2_qe_base)

    # Late-stage score interpolation combines expanded retrievers while bypassing Java integer docid casting limitations
    best_bm25_qe = bm25_qe_base >> pt.rewrite.Bo1QueryExpansion(index, fb_docs=5, fb_terms=20) >> bm25_qe_base
    best_inexp_qe = inexpb2_qe_base >> pt.rewrite.Bo1QueryExpansion(index, fb_docs=3, fb_terms=10) >> inexpb2_qe_base
    for alpha in [0.2, 0.3, 0.5, 0.7]:
        p3_names.append(f"Hybrid_Expanded_a{alpha}")
        p3_pipes.append(alpha * best_bm25_qe + (1 - alpha) * best_inexp_qe)

    res3, mc3 = run_experiment(
        p3_pipes, p3_names, queries_df, qrels_df,
        baseline_name=p3_names[0],
        section_label="PART 3 -- Query Expansion on Best BM25 & In_expB2 + Expanded Hybrid"
    )
    res3["section"] = "Part3_QE"
    all_section_results.append(res3)

    combined = pd.concat(all_section_results, ignore_index=True)
    out_file = f"{GROUP_PREFIX}_eval_results_extended.txt"
    combined.to_csv(out_file, index=False)
    print(f"\n{'=' * 65}")
    print(f"Results saved to '{out_file}'")

    map_col = "MAP" if "MAP" in combined.columns else "AP"
    best_overall = combined.sort_values(by=map_col, ascending=False).iloc[0]
    print(f"\nOverall Best Extended Model: {best_overall['name']}  "
          f"(Section: {best_overall['section']}, MAP = {best_overall[map_col]:.4f})")

    best_sec = best_overall["section"]
    if best_sec == "Part2_BM25_DFR_Hybrid":
        best_pipe = p2_pipes[p2_names.index(best_overall["name"])]
    else:
        best_pipe = p3_pipes[p3_names.index(best_overall["name"])]

    print(f"\nGenerating TREC run for '{best_overall['name']}' ...")
    best_run = best_pipe.transform(queries_df)
    extended_results_file = f"{GROUP_PREFIX}_results_extended.txt"
    save_trec_run(best_run, extended_results_file, run_tag="info_fetch_extended")
    print(f"Extended best run saved to '{extended_results_file}'.")
    print("\nExtended evaluation complete.")


if __name__ == "__main__":
    main()
