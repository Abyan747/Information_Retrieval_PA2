# Programming Assignment II: Vector Space Model & Ranked Retrieval
**Information Retrieval (CS60092) — Autumn 2026–27**  
**Department of Computer Science and Engineering, IIT Kharagpur**

---

## 1. Group Details

* **Group Name:** Info fetch
* **Repository:** `https://github.com/Abyan747/Information_Retrieval_PA2`
* **Team Members:**
  * **Sumit Mukherjee** — `23EC3AI17` (Dual Degree, E&ECE)
  * **Abyan Hussain** — `23EC3AI19` (Dual Degree, E&ECE)
  * **Kinnar Halder** — `23IE35001` (Dual Degree, Industrial & Systems)
  * **Avik Ghosh** — `23EC10099` (B.Tech, E&ECE)

---

## 2. Overview & Architecture

This repository implements, tunes, and evaluates ranked retrieval pipelines on the Cranfield text collection (1,400 documents, 225 test queries, 1,612 human relevance judgments).

The codebase conforms strictly to the assignment's core constraint:
> *"You may use only sparse vector space models. Don't use advanced probabilistic or dense neural retrieval models."*

The repository is organized across four retrieval paradigms:
1. **Part I — Pure Sparse Vector Space Models (Primary Submission):**
   * High-resolution 17-point sweep over the document length normalization parameter $c \in [0.1, 2.0]$ in pivoted `TF_IDF`.
   * Baseline models: Raw Term Frequency (`Tf`), Coordinate Matching (`CoordinateMatch`), and `LemurTF_IDF`.
   * **Peak Model:** `TF_IDF_c0.85` ($\text{MAP} = \mathbf{0.3200}$, $\text{nDCG@10} = \mathbf{0.3486}$).
2. **Part II — Pure VSM with Pseudo-Relevance Feedback (PRF):**
   * Two-pass query expansion on `TF_IDF_c0.85` using Bose-Einstein (`Bo1`) and Kullback-Leibler (`KL`) divergence.
   * **Peak Model:** `VSM_Bo1_d3_t25` ($\text{MAP} = \mathbf{0.3513}$, $\text{nDCG@10} = \mathbf{0.3789}$, $+9.77\%$ MAP gain).
3. **Part III — Advanced Probabilistic & DFR Models (Report Comparison):**
   * Two-dimensional parameter sweep over BM25 ($b \in [0.3, 1.0]$, $k_1 \in [1.0, 2.0]$).
   * Divergence from Randomness (DFR) sweep on `In_expB2` ($c \in [0.1, 2.0]$), `PL2`, `BB2`, `DPH`, and `DFRee`.
   * Best-of-both linear score interpolation hybrid: $\alpha \cdot \text{BM25}^* + (1 - \alpha) \cdot \text{InExpB2}^*$.
   * **Peak Model:** `In_expB2_c0.7` ($\text{MAP} = \mathbf{0.3312}$).
4. **Part IV — Query Expansion on Advanced Models & Expanded Ensembles:**
   * PRF applied to the tuned BM25 and In_expB2 models.
   * Late-stage score-interpolated ensemble ($0.5 \cdot \text{BM25}_{\text{QE}} + 0.5 \cdot \text{InExp}_{\text{QE}}$).
   * **Global Peak:** `Hybrid_Expanded_a0.5` ($\text{MAP} = \mathbf{0.3534}$, $\text{P@5} = \mathbf{0.3556}$).

---

## 3. Environment & Prerequisites

* **Python:** 3.8+ (tested on Python 3.10, 3.11, and 3.13)
* **Java:** Java 11 or higher (JRE/JDK path required for PyTerrier / PyJNIus JVM bridge)

### Installation
```bash
pip install python-terrier ir_measures pandas nltk
```

*Note on Java Configuration:* If `JAVA_HOME` is not set in your environment variables, [`config.py`](file:///c:/IITKGP/IR/PA2/config.py) includes automated detection to ensure PyTerrier initializes without manual environment configuration.

---

## 4. Execution Workflow

All steps can be executed sequentially from the terminal:

### Step 1: Preprocessing
Extracts Title (`.T`) and Abstract (`.W`) fields, removes 358 stopwords, and applies Porter stemming:
```bash
python info_fetch_preprocess.py
```
*Outputs:*
* `info_fetch_processed.all` — Stemmed corpus
* `info_fetch_processed_queries.txt` — Stemmed queries

---

### Step 2: Index Construction
Builds an inverted index via PyTerrier with whitespace tokenization:
```bash
python info_fetch_index.py
```
*Outputs:*
* `info_fetch_terrier_index/` — Directory containing the inverted index
* `info_fetch_index_stats.txt` — Index statistics (1,400 docs, 4,188 terms, 77,782 postings)

---

### Step 3: Ranked Retrieval (Search)

#### A. Default Execution (Runs Best Pure VSM Model: `TF_IDF_c0.85`)
```bash
python info_fetch_search.py
```
*Outputs:*
* `info_fetch_results.txt` — Official submission TREC run file (189,786 rows).

#### B. Search on Custom / Unknown Test Queries (For Automated Grading)
To evaluate on a new or hidden query file using the default compliant Pure VSM model (`TF_IDF_c0.85`):
```bash
python info_fetch_search.py <path_to_queries> [path_to_output_run]
```
*Example:*
```bash
python info_fetch_search.py custom_test_queries.txt custom_run.txt
```

#### C. Running Hidden Queries on Other Models (`--model`)
You can evaluate hidden test queries on **any** of the retrieval models using the `--model` flag:
```bash
python info_fetch_search.py <path_to_queries> [path_to_output_run] --model <model_name>
```

| Model Choice | Paradigm | Command Example | Performance on Cranfield |
|---|---|---|:---:|
| `vsm` *(default)* | Pure Sparse VSM (`TF_IDF_c0.85`) | `python info_fetch_search.py hidden_queries.txt run.txt --model vsm` | MAP = **0.3200** |
| `vsm_qe` | VSM + Bo1 Query Expansion | `python info_fetch_search.py hidden_queries.txt run.txt --model vsm_qe` | MAP = **0.3513** |
| `inexpb2` | Tuned In_expB2 DFR ($c=0.7$) | `python info_fetch_search.py hidden_queries.txt run.txt --model inexpb2` | MAP = **0.3312** |
| `bm25` | Tuned BM25 ($b=0.75, k_1=2.0$) | `python info_fetch_search.py hidden_queries.txt run.txt --model bm25` | MAP = **0.3217** |
| `hybrid` | Best-of-Both Hybrid ($0.1\text{BM25} + 0.9\text{DFR}$) | `python info_fetch_search.py hidden_queries.txt run.txt --model hybrid` | MAP = **0.3302** |
| `hybrid_qe` | Expanded Hybrid ($0.5\text{BM25}_{\text{QE}} + 0.5\text{DFR}_{\text{QE}}$) | `python info_fetch_search.py hidden_queries.txt run.txt --model hybrid_qe` | MAP = **0.3534** |

#### D. Full Experimental Suite & Latency Profiling
To run all 20 single-pass and PRF models to measure per-query execution latencies:
```bash
python info_fetch_search.py --experiments
```
*Outputs:*
* `info_fetch_latency.txt` — Latency table in milliseconds per query
* `info_fetch_runs/*.txt` — Individual TREC run files for every evaluated model

---

### Step 4: Evaluation Scripts

#### A. Primary Submission Evaluation — Pure Sparse VSM (`info_fetch_evaluate.py`)
Evaluates the 20 pure sparse VSM pipelines against `cranqrel` with paired significance tests:
```bash
python info_fetch_evaluate.py
```
*Outputs:*
* `info_fetch_eval_results.txt` — Full evaluation table with MAP, nDCG@10, P@5, P@10, R@100, and $p$-values.
* Updates `info_fetch_results.txt` with the ranked run of the optimal model (`TF_IDF_c0.85`).

#### B. Pure VSM + Query Expansion (`evaluate_query_expansion_with_VSM.py`)
Evaluates Bose-Einstein (Bo1) and Kullback-Leibler (KL) PRF on the optimal VSM:
```bash
python evaluate_query_expansion_with_VSM.py
```
*Outputs:*
* `eval_query_expansion_vsm_results.txt` — 29-pipeline PRF evaluation table and significance metrics.
* `results_query_expansion_vsm.txt` — TREC run file for `VSM_Bo1_d3_t25` ($\text{MAP} = 0.3513$).

#### C. Advanced Models & Hybrids (`info_fetch_evaluate_extended.py`)
Evaluates BM25 tuning ($b, k_1$), In_expB2 tuning ($c$), DFR baselines, Best-of-Both Hybrid interpolation, and Advanced PRF:
```bash
python info_fetch_evaluate_extended.py
```
*Outputs:*
* `info_fetch_eval_results_extended.txt` — Evaluation table across all 67 advanced configurations.
* `info_fetch_results_extended.txt` — TREC run file for `Hybrid_Expanded_a0.5` ($\text{MAP} = 0.3534$).

---

## 5. File Manifest

| File Name | Purpose |
|---|---|
| `config.py` | Central configuration, file paths, and automated Java environment setup |
| `info_fetch_preprocess.py` | State-machine SGML parser and 4-stage preprocessing pipeline |
| `info_fetch_utils.py` | Preprocessing utilities (tokenizer regex, stopword filter, Porter stemmer) |
| `info_fetch_index.py` | PyTerrier inverted index builder with whitespace tokenizer |
| `info_fetch_search.py` | Primary retrieval engine supporting CLI query arguments and latency profiling |
| `info_fetch_evaluate.py` | **Official Submission Script:** Evaluates Pure Sparse VSM pipelines |
| `evaluate_query_expansion_with_VSM.py` | Evaluates Pseudo-Relevance Feedback (Bo1/KL) on the optimal pure VSM |
| `info_fetch_evaluate_extended.py` | Evaluates advanced models: BM25, In_expB2, Hybrids, and Advanced PRF |
| `info_fetch_results.txt` | **Primary Submission File:** Standard 6-column TREC run of the optimal pure VSM |


---

## 6. Experimental Results Summary

| Paradigm Tier | Optimal Model Configuration | MAP | nDCG@10 | P@5 | P@10 | Recall@100 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Part I: Pure Sparse VSM** | `TF_IDF_c0.85` | **0.3200** | **0.3486** | 0.3271 | 0.2364 | 0.7502 |
| **Part II: VSM + PRF** | `VSM_Bo1_d3_t25` | **0.3513** | **0.3789** | 0.3547 | 0.2716 | 0.7875 | 
| **Part III: Advanced Models** | `In_expB2_c0.7` | **0.3312** | **0.3645** | 0.3396 | 0.2573 | 0.7651 | 
| **Part III: Tuned BM25** | `BM25_b0.75_k1_2.0` | 0.3217 | 0.3522 | 0.3342 | 0.2462 | 0.7493 |
| **Part IV: Expanded Hybrid** | `Hybrid_Expanded_a0.5` | **0.3534** | 0.3793 | **0.3556** | 0.2693 | 0.7905 |

---

## 7. Submission Package Structure

To submit on Moodle, archive the project directory as a `.zip` file:
```text
info_fetch_PA2.zip
├── info_fetch_results.txt
├── info_fetch_report.pdf (compiled from report.tex)
├── config.py
├── info_fetch_preprocess.py
├── info_fetch_utils.py
├── info_fetch_index.py
├── info_fetch_search.py
├── info_fetch_evaluate.py
├── evaluate_query_expansion_with_VSM.py
├── info_fetch_evaluate_extended.py
└── README.md
```
