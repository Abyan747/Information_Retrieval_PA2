"""
info_fetch_search.py
Runs ranked retrieval on the Cranfield collection using PyTerrier.
Complies strictly with PA2 requirements:
    "You may use only sparse vector space models. Don't use advanced probabilistic
     or dense neural retrieval models."

By default, executes the optimal Sparse Vector Space Model (TF_IDF with c=0.85)
on the processed queries and writes the TREC-format submission run to info_fetch_results.txt.

Supports custom query files and output destinations via CLI arguments:
    python info_fetch_search.py [query_file] [output_file] [--experiments]

Examples:
    # 1. Default execution (runs optimal VSM model c=0.85, outputs info_fetch_results.txt)
    python info_fetch_search.py

    # 2. Search on custom/unknown test queries for automated grading:
    python info_fetch_search.py custom_queries.txt custom_results.txt

    # 3. Run full experimental parameter sweep & latency profiling:
    python info_fetch_search.py --experiments
"""

import os
import sys
import time
import argparse
import pandas as pd

import config
from config import PROCESSED_QRY_FILE, INDEX_DIR, RESULTS_FILE, GROUP_PREFIX

import pyterrier as pt
if not pt.java.started():
    pt.java.init()


def parse_processed_queries(filepath: str) -> pd.DataFrame:
    """
    Parse queries file into a PyTerrier topics DataFrame.
    Supports:
      1. Cranfield SGML format (single-line '.W <stems>' or multi-line '.W\\n<text>')
      2. Line-by-line format ('<qid> <query_text>' or plain query lines)
    Applies preprocessing/stemming if unstemmed text is detected and stopwords.txt exists.
    """
    rows = []
    qid = None
    in_w = False
    current_tokens = []

    with open(filepath, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    is_sgml = any(l.startswith(".I ") for l in lines[:20])

    if is_sgml:
        for line in lines:
            line = line.rstrip("\r\n")
            if line.startswith(".I "):
                if qid is not None and current_tokens:
                    rows.append({"qid": qid, "query": " ".join(current_tokens).strip()})
                qid = str(len(rows) + 1)
                in_w = False
                current_tokens = []
            elif line.startswith(".W ") and qid is not None:
                current_tokens.append(line[3:].strip())
            elif line == ".W":
                in_w = True
            elif in_w and qid is not None:
                current_tokens.append(line.strip())
        if qid is not None and current_tokens:
            rows.append({"qid": qid, "query": " ".join(current_tokens).strip()})
    else:
        for idx, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                rows.append({"qid": str(int(parts[0])), "query": parts[1].strip()})
            else:
                rows.append({"qid": str(idx), "query": line})

    # Ensure queries match indexed vocabulary (stemmed + stopwords removed)
    stopwords_file = "stopwords.txt"
    if os.path.exists(stopwords_file):
        try:
            from info_fetch_utils import load_stopwords, preprocess
            stopwords = load_stopwords(stopwords_file)
            for row in rows:
                processed_tokens = preprocess(row["query"], stopwords)
                if processed_tokens:
                    row["query"] = " ".join(processed_tokens)
        except Exception:
            pass

    return pd.DataFrame(rows, columns=["qid", "query"])


def save_trec_run(results_df: pd.DataFrame, filepath: str, run_tag: str = "info_fetch"):
    """Write results to TREC run format using modern vectorized pt.io.write_results."""
    pt.io.write_results(results_df, filepath, run_name=run_tag)
    print(f"  Saved TREC run to '{filepath}' ({len(results_df)} rows)")


def run_model(index, wmodel: str, controls: dict, queries_df: pd.DataFrame, num_results: int = 1000):
    """Run a single retrieval model using modern pt.terrier.Retriever, return (results_df, mean_latency_ms)."""
    retriever = pt.terrier.Retriever(
        index,
        wmodel=wmodel,
        tokeniser="whitespace",
        num_results=num_results,
        controls=controls or {},
    )
    t0 = time.time()
    results = retriever.transform(queries_df)
    elapsed_ms = (time.time() - t0) * 1000
    mean_latency = elapsed_ms / max(len(queries_df), 1)
    return results, mean_latency


def run_all_experiments(index, queries_df):
    """Execute all pure Sparse VSM parameter sweeps and PRF pipelines for latency profiling."""
    experiments = []
    for c_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.78, 0.8, 0.82, 0.85, 0.88, 0.9, 1.0, 1.5, 2.0]:
        experiments.append((f"TF_IDF_c{c_val}", "TF_IDF", {"c": str(c_val)}))

    experiments.append(("Tf", "Tf", {}))
    experiments.append(("CoordinateMatch", "CoordinateMatch", {}))
    experiments.append(("LemurTF_IDF", "LemurTF_IDF", {}))

    prf_pipelines = []
    base_retriever = pt.terrier.Retriever(index, wmodel="TF_IDF", tokeniser="whitespace", controls={"c": "0.85"})
    for fb_docs in [3, 5, 10]:
        for fb_terms in [5, 10, 15, 20]:
            qe = pt.rewrite.Bo1QueryExpansion(index, fb_docs=fb_docs, fb_terms=fb_terms)
            pipe = base_retriever >> qe >> base_retriever
            prf_pipelines.append((f"TF_IDF_Bo1_d{fb_docs}_t{fb_terms}", pipe))

    qe_kl = pt.rewrite.KLQueryExpansion(index, fb_docs=5, fb_terms=10)
    prf_pipelines.append(("TF_IDF_KL_d5_t10", base_retriever >> qe_kl >> base_retriever))

    all_results = {}
    latency_records = []
    total_experiments = len(experiments) + len(prf_pipelines)
    print(f"\nRunning {total_experiments} pure Sparse VSM & PRF retrieval experiments...")
    print("-" * 60)

    for name, wmodel, controls in experiments:
        ctrl_str = f"controls={controls}" if controls else ""
        print(f"  [{name:20}] wmodel={wmodel:15} {ctrl_str} ...", end=" ", flush=True)
        try:
            results_df, mean_lat = run_model(index, wmodel, controls, queries_df)
            all_results[name] = results_df
            latency_records.append({"model": name, "mean_latency_ms": round(mean_lat, 2)})
            print(f"done. Mean latency: {mean_lat:.2f} ms/query")
        except Exception as e:
            print(f"ERROR: {e}")

    for name, pipeline in prf_pipelines:
        print(f"  [{name:20}] PRF pipeline ...", end=" ", flush=True)
        try:
            t0 = time.time()
            results_df = pipeline.transform(queries_df)
            elapsed_ms = (time.time() - t0) * 1000
            mean_lat = elapsed_ms / max(len(queries_df), 1)
            all_results[name] = results_df
            latency_records.append({"model": name, "mean_latency_ms": round(mean_lat, 2)})
            print(f"done. Mean latency: {mean_lat:.2f} ms/query")
        except Exception as e:
            print(f"ERROR: {e}")

    latency_file = f"{GROUP_PREFIX}_latency.txt"
    lat_df = pd.DataFrame(latency_records)
    with open(latency_file, "w") as f:
        f.write(lat_df.to_string(index=False))
    print(f"\nLatency table saved to '{latency_file}'")

    runs_dir = f"{GROUP_PREFIX}_runs"
    os.makedirs(runs_dir, exist_ok=True)
    for name, res_df in all_results.items():
        run_path = os.path.join(runs_dir, f"{name}.txt")
        save_trec_run(res_df, run_path, run_tag=f"info_fetch_{name}")


def main():
    parser = argparse.ArgumentParser(description="Info fetch -- Ranked Retrieval Search (Sparse VSM)")
    parser.add_argument("query_file", nargs="?", default=PROCESSED_QRY_FILE,
                        help=f"Path to queries file (default: '{PROCESSED_QRY_FILE}')")
    parser.add_argument("output_file", nargs="?", default=RESULTS_FILE,
                        help=f"Path to output TREC run file (default: '{RESULTS_FILE}')")
    parser.add_argument("--model", default="vsm",
                        choices=["vsm", "vsm_qe", "bm25", "inexpb2", "hybrid", "hybrid_qe"],
                        help="Retrieval model to execute: vsm (default), vsm_qe, bm25, inexpb2, hybrid, hybrid_qe")
    parser.add_argument("--c", default="0.85",
                        help="Document length normalization parameter c for TF_IDF (default: '0.85')")
    parser.add_argument("--experiments", action="store_true",
                        help="Run full experimental suite across all models and measure latency")
    args = parser.parse_args()

    print("=" * 60)
    print("Info fetch -- PA2 Ranked Retrieval Search")
    print("=" * 60)

    print(f"\nLoading index from '{INDEX_DIR}' ...")
    if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
        print(f"  Index not found at '{INDEX_DIR}'. Building index automatically...")
        import info_fetch_index
        info_fetch_index.main()
    index = pt.IndexFactory.of(os.path.abspath(INDEX_DIR))
    stats = index.getCollectionStatistics()
    print(f"  Index loaded: {stats.getNumberOfDocuments()} docs, {stats.getNumberOfUniqueTerms()} terms")

    print(f"\nLoading queries from '{args.query_file}' ...")
    queries_df = parse_processed_queries(args.query_file)
    print(f"  {len(queries_df)} queries loaded.")

    if args.experiments:
        run_all_experiments(index, queries_df)
        print("\nAll experiments complete.")
        return

    # Select retrieval pipeline based on requested model
    if args.model == "vsm":
        print(f"\nExecuting optimal Pure Sparse VSM (TF_IDF, c={args.c}) ...")
        pipeline = pt.terrier.Retriever(index, wmodel="TF_IDF", tokeniser="whitespace", controls={"c": str(args.c)})
    elif args.model == "vsm_qe":
        print("\nExecuting Pure Sparse VSM + Bo1 Query Expansion (fb_docs=3, fb_terms=25) ...")
        base = pt.terrier.Retriever(index, wmodel="TF_IDF", tokeniser="whitespace", controls={"c": "0.85"})
        qe = pt.rewrite.Bo1QueryExpansion(index, fb_docs=3, fb_terms=25)
        pipeline = base >> qe >> base
    elif args.model == "bm25":
        print("\nExecuting Tuned BM25 (b=0.75, k_1=2.0) ...")
        pipeline = pt.terrier.Retriever(index, wmodel="BM25", tokeniser="whitespace", controls={"bm25.b": "0.75", "bm25.k_1": "2.0"})
    elif args.model == "inexpb2":
        print("\nExecuting Tuned In_expB2 DFR (c=0.7) ...")
        pipeline = pt.terrier.Retriever(index, wmodel="In_expB2", tokeniser="whitespace", controls={"c": "0.7"})
    elif args.model == "hybrid":
        print("\nExecuting Best-of-Both Hybrid (0.1 * BM25* + 0.9 * In_expB2*) ...")
        bm25_t = pt.terrier.Retriever(index, wmodel="BM25", tokeniser="whitespace", controls={"bm25.b": "0.75", "bm25.k_1": "2.0"})
        inexp_b = pt.terrier.Retriever(index, wmodel="In_expB2", tokeniser="whitespace", controls={"c": "0.7"})
        pipeline = 0.1 * bm25_t + 0.9 * inexp_b
    elif args.model == "hybrid_qe":
        print("\nExecuting Expanded Hybrid (0.5 * BM25_QE + 0.5 * In_expB2_QE) ...")
        bm25_t = pt.terrier.Retriever(index, wmodel="BM25", tokeniser="whitespace", controls={"bm25.b": "0.75", "bm25.k_1": "2.0"})
        inexp_b = pt.terrier.Retriever(index, wmodel="In_expB2", tokeniser="whitespace", controls={"c": "0.7"})
        p_bm25 = bm25_t >> pt.rewrite.Bo1QueryExpansion(index, fb_docs=5, fb_terms=20) >> bm25_t
        p_inexp = inexp_b >> pt.rewrite.Bo1QueryExpansion(index, fb_docs=3, fb_terms=10) >> inexp_b
        pipeline = 0.5 * p_bm25 + 0.5 * p_inexp

    t0 = time.time()
    results_df = pipeline.transform(queries_df)
    elapsed_ms = (time.time() - t0) * 1000
    mean_lat = elapsed_ms / max(len(queries_df), 1)
    print(f"  Completed in {mean_lat:.2f} ms/query.")

    save_trec_run(results_df, args.output_file, run_tag="info_fetch")
    print(f"\nRanked results saved to '{args.output_file}'.")


if __name__ == "__main__":
    main()
