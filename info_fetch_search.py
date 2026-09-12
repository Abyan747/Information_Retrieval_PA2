"""
info_fetch_search.py
Runs ranked retrieval experiments on the Cranfield collection using PyTerrier.
Experiments with multiple sparse VSM weighting models and parameter sweeps.

Models tested:
  - TF_IDF  (classic TF-IDF cosine similarity)
  - BM25    (b sweep 0.1-0.9)
  - PL2     (c sweep 0.1-0.9)
  - In_expB2 (DFR variant)
  - DLH     (DFR hypergeometric)

Outputs:
  - info_fetch_results.txt  : TREC-format ranked results for best model
  - info_fetch_latency.txt  : per-model mean query latency

Usage:
    python info_fetch_search.py
"""

import os
import time
import pandas as pd

from config import PROCESSED_QRY_FILE, INDEX_DIR, RESULTS_FILE, GROUP_PREFIX
import pyterrier as pt


def parse_processed_queries(filepath: str) -> pd.DataFrame:
    """
    Parse info_fetch_processed_queries.txt into a PyTerrier topics DataFrame.
    Expected format:
        .I <qid>
        .W <stem1> <stem2> ...
    Returns DataFrame with columns: qid (str), query (str)
    """
    rows = []
    qid = None
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith(".I "):
                # Cranfield relevance judgments (cranqrel) index queries 1..225 sequentially
                qid = str(len(rows) + 1)
            elif line.startswith(".W ") and qid is not None:
                query = line[3:].strip()
                if query:
                    rows.append({"qid": qid, "query": query})
                qid = None
    return pd.DataFrame(rows, columns=["qid", "query"])


def save_trec_run(results_df: pd.DataFrame, filepath: str, run_tag: str = "info_fetch"):
    """Write results to TREC run file format."""
    with open(filepath, "w") as f:
        for _, row in results_df.iterrows():
            f.write(f"{row['qid']} Q0 {row['docno']} {int(row['rank'])} {row['score']:.6f} {run_tag}\n")
    print(f"  Saved TREC run to '{filepath}' ({len(results_df)} rows)")


def run_model(index, wmodel, controls, queries_df, num_results=1000):
    """Run a single retrieval model, return (results_df, mean_latency_ms)."""
    retriever = pt.BatchRetrieve(
        index,
        wmodel=wmodel,
        num_results=num_results,
        controls=controls,
    )
    t0 = time.time()
    results = retriever.transform(queries_df)
    elapsed_ms = (time.time() - t0) * 1000
    mean_latency = elapsed_ms / len(queries_df)
    return results, mean_latency


def main():
    print("=" * 60)
    print("Info fetch -- PA2 Search & Retrieval Experiments")
    print("=" * 60)

    # Load index
    print(f"\nLoading index from '{INDEX_DIR}' ...")
    index = pt.IndexFactory.of(os.path.abspath(INDEX_DIR))
    print(f"  Index loaded: {index.getCollectionStatistics().getNumberOfDocuments()} docs, "
          f"{index.getCollectionStatistics().getNumberOfUniqueTerms()} terms")

    # Load queries
    print(f"\nLoading queries from '{PROCESSED_QRY_FILE}' ...")
    queries_df = parse_processed_queries(PROCESSED_QRY_FILE)
    print(f"  {len(queries_df)} queries loaded.")

    # ----------------------------------------------------------------
    # Define all experiments
    # ----------------------------------------------------------------
    experiments = []

    # 1. Baseline TF_IDF
    experiments.append(("TF_IDF", "TF_IDF", {}))

    # 2. BM25 b sweep
    for b in [0.1, 0.3, 0.5, 0.75, 0.9]:
        experiments.append((f"BM25_b{b}", "BM25", {"bm25.b": b, "bm25.k_1": 1.2, "bm25.k_3": 8}))

    # 3. PL2 c sweep
    for c in [0.1, 0.3, 0.5, 0.75, 1.0]:
        experiments.append((f"PL2_c{c}", "PL2", {"c": c}))

    # 4. Other DFR models
    experiments.append(("In_expB2", "In_expB2", {}))

    # ----------------------------------------------------------------
    # Run all experiments
    # ----------------------------------------------------------------
    all_results = {}
    latency_records = []

    print(f"\nRunning {len(experiments)} retrieval experiments...")
    print("-" * 60)

    for name, wmodel, controls in experiments:
        print(f"  [{name}] wmodel={wmodel}, controls={controls} ...", end=" ", flush=True)
        try:
            results_df, mean_lat = run_model(index, wmodel, controls, queries_df)
            all_results[name] = results_df
            latency_records.append({"model": name, "mean_latency_ms": round(mean_lat, 2)})
            print(f"done. Mean latency: {mean_lat:.1f} ms/query")
        except Exception as e:
            print(f"ERROR: {e}")

    # ----------------------------------------------------------------
    # Save latency table
    # ----------------------------------------------------------------
    latency_file = f"{GROUP_PREFIX}_latency.txt"
    lat_df = pd.DataFrame(latency_records)
    with open(latency_file, "w") as f:
        f.write(lat_df.to_string(index=False))
    print(f"\nLatency table saved to '{latency_file}'")

    # ----------------------------------------------------------------
    # Save all individual run files (for evaluation)
    # ----------------------------------------------------------------
    runs_dir = f"{GROUP_PREFIX}_runs"
    os.makedirs(runs_dir, exist_ok=True)
    for name, res_df in all_results.items():
        run_path = os.path.join(runs_dir, f"{name}.txt")
        save_trec_run(res_df, run_path, run_tag=f"info_fetch_{name}")

    # ----------------------------------------------------------------
    # Save default best run as info_fetch_results.txt (TF_IDF as default)
    # Will be overwritten by evaluate.py with actual best model
    # ----------------------------------------------------------------
    if "TF_IDF" in all_results:
        save_trec_run(all_results["TF_IDF"], RESULTS_FILE, run_tag="info_fetch")
        print(f"\nDefault results.txt written (TF_IDF). Run info_fetch_evaluate.py to pick best model.")

    print("\nDone.")


if __name__ == "__main__":
    main()
