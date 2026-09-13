"""
info_fetch_search.py
Runs ranked retrieval experiments on the Cranfield collection using PyTerrier.
Complies strictly with PA2 requirements:
    "You may use only sparse vector space models. Don't use advanced probabilistic
     or dense neural retrieval models."

Sparse Vector Space Models & Parameter Sweeps Tested:
  1. TF_IDF (Classic Salton/Robertson VSM with length normalization parameter c):
     - Sweeping c across [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.5, 2.0]
  2. Tf (Raw Term Frequency baseline, no IDF, no length normalization)
  3. CoordinateMatch (Coordinate matching baseline, count of matching query terms)
  4. LemurTF_IDF (Lemur Vector Space TF-IDF formulation)

Modern PyTerrier optimizations:
  - Uses pt.terrier.Retriever() (replaces deprecated pt.BatchRetrieve)
  - tokeniser="whitespace" (preserves preprocessed stems)
  - Uses vectorized pt.io.write_results() (replaces manual iterrows() loops)

Outputs:
  - info_fetch_results.txt : TREC-format ranked results for default/best model
  - info_fetch_latency.txt : Per-model mean query latency in ms
  - info_fetch_runs/*.txt  : TREC-format runs for all individual models

Usage:
    python info_fetch_search.py
"""

import os
import time
import pandas as pd

from config import PROCESSED_QRY_FILE, INDEX_DIR, RESULTS_FILE, GROUP_PREFIX

import pyterrier as pt
if not pt.java.started():
    pt.java.init()


def parse_processed_queries(filepath: str) -> pd.DataFrame:
    """
    Parse info_fetch_processed_queries.txt into a PyTerrier topics DataFrame.
    Expected format:
        .I <qid>
        .W <stem1> <stem2> ...
    Returns DataFrame with columns: qid (str), query (str).
    Queries are indexed 1..225 sequentially to match cranqrel.
    """
    rows = []
    qid = None
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith(".I "):
                qid = str(len(rows) + 1)
            elif line.startswith(".W ") and qid is not None:
                query = line[3:].strip()
                if query:
                    rows.append({"qid": qid, "query": query})
                qid = None
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


def main():
    print("=" * 60)
    print("Info fetch -- PA2 Sparse Vector Space Model Experiments")
    print("=" * 60)

    # Load index
    print(f"\nLoading index from '{INDEX_DIR}' ...")
    index = pt.IndexFactory.of(os.path.abspath(INDEX_DIR))
    stats = index.getCollectionStatistics()
    print(f"  Index loaded: {stats.getNumberOfDocuments()} docs, {stats.getNumberOfUniqueTerms()} terms")

    # Load queries
    print(f"\nLoading queries from '{PROCESSED_QRY_FILE}' ...")
    queries_df = parse_processed_queries(PROCESSED_QRY_FILE)
    print(f"  {len(queries_df)} queries loaded.")

    # ----------------------------------------------------------------
    # Define Sparse Vector Space Model Experiments
    # ----------------------------------------------------------------
    experiments = []

    # 1. TF_IDF document length normalization parameter sweep (c)
    for c_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.5, 2.0]:
        experiments.append((f"TF_IDF_c{c_val}", "TF_IDF", {"c": str(c_val)}))

    # 2. Raw Term Frequency baseline (no IDF, no normalization)
    experiments.append(("Tf", "Tf", {}))

    # 3. Coordinate Matching baseline (term overlap count)
    experiments.append(("CoordinateMatch", "CoordinateMatch", {}))

    # 4. Lemur TF-IDF Vector Space formulation
    experiments.append(("LemurTF_IDF", "LemurTF_IDF", {}))

    # ----------------------------------------------------------------
    # Execute All Experiments
    # ----------------------------------------------------------------
    all_results = {}
    latency_records = []

    print(f"\nRunning {len(experiments)} pure Sparse VSM retrieval experiments...")
    print("-" * 60)

    for name, wmodel, controls in experiments:
        ctrl_str = f"controls={controls}" if controls else ""
        print(f"  [{name:16}] wmodel={wmodel:15} {ctrl_str} ...", end=" ", flush=True)
        try:
            results_df, mean_lat = run_model(index, wmodel, controls, queries_df)
            all_results[name] = results_df
            latency_records.append({"model": name, "mean_latency_ms": round(mean_lat, 2)})
            print(f"done. Mean latency: {mean_lat:.2f} ms/query")
        except Exception as e:
            print(f"ERROR: {e}")

    # ----------------------------------------------------------------
    # Save Latency Table
    # ----------------------------------------------------------------
    latency_file = f"{GROUP_PREFIX}_latency.txt"
    lat_df = pd.DataFrame(latency_records)
    with open(latency_file, "w") as f:
        f.write(lat_df.to_string(index=False))
    print(f"\nLatency table saved to '{latency_file}'")

    # ----------------------------------------------------------------
    # Save All Individual Run Files
    # ----------------------------------------------------------------
    runs_dir = f"{GROUP_PREFIX}_runs"
    os.makedirs(runs_dir, exist_ok=True)
    for name, res_df in all_results.items():
        run_path = os.path.join(runs_dir, f"{name}.txt")
        save_trec_run(res_df, run_path, run_tag=f"info_fetch_{name}")

    # ----------------------------------------------------------------
    # Save Default Run (TF_IDF_c0.75) as Submission File
    # Will be verified / finalized by info_fetch_evaluate.py
    # ----------------------------------------------------------------
    default_best = "TF_IDF_c0.75" if "TF_IDF_c0.75" in all_results else list(all_results.keys())[0]
    save_trec_run(all_results[default_best], RESULTS_FILE, run_tag="info_fetch")
    print(f"\nDefault results.txt written ({default_best}). Run info_fetch_evaluate.py to evaluate models.")

    print("\nSearch experiments complete.")


if __name__ == "__main__":
    main()
