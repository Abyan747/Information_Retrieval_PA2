"""
config.py
Central configuration for the Cranfield Ranked Retrieval IR system (PA2).
Group Name: Info fetch
"""

import os

# PyJNIus requires an explicit JRE home when system environment variables lack JAVA_HOME
if "JAVA_HOME" not in os.environ or not os.path.exists(os.environ.get("JAVA_HOME", "")):
    candidate_java = r"C:\Users\abyan\AppData\Local\JDownloader 2\jre"
    if os.path.exists(candidate_java):
        os.environ["JAVA_HOME"] = candidate_java

GROUP_NAME   = "Info fetch"
GROUP_PREFIX = "info_fetch"     # filesystem-safe: lowercase, spaces -> underscores

CORPUS_FILE    = "cran.all.1400" if os.path.exists("cran.all.1400") else "cran.all"
STOPWORDS_FILE = "stopwords.txt"
QUERY_FILE     = "cran.qry"
REL_FILE       = "cranqrel"

PROCESSED_FILE = f"{GROUP_PREFIX}_processed.all"
PROCESSED_QRY_FILE = f"{GROUP_PREFIX}_processed_queries.txt"
RESULTS_FILE   = f"{GROUP_PREFIX}_results.txt"
INDEX_DIR      = f"{GROUP_PREFIX}_terrier_index"
