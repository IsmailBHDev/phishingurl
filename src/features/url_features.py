# src/features/url_features.py

import re
import math
import pandas as pd
from urllib.parse import urlparse
from collections import Counter


# -------------------------
# Constants
# -------------------------

SUSPICIOUS_WORDS = [
    "login", "verify", "account", "secure", "update",
    "bank", "free", "confirm", "signin", "password"
]

SUSPICIOUS_TLDS = {".xyz", ".top", ".ru", ".cn", ".tk", ".ml", ".ga", ".cf"}

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd"}


# -------------------------
# Helpers
# -------------------------

def entropy(s: str) -> float:
    """Compute Shannon entropy of a string."""
    if not s:
        return 0.0
    counts = Counter(s)
    probs = [c / len(s) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def has_ip_address(hostname: str) -> int:
    """Check if hostname is an IP address."""
    return int(bool(re.fullmatch(r"\d+\.\d+\.\d+\.\d+", hostname)))


# -------------------------
# Feature extraction
# -------------------------

def extract_url_features(url: str) -> dict:
    """Extract phishing-related features from a URL."""
    features = {}

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
    except Exception:
        hostname = ""
        path = ""
        query = ""

    # Length-based
    features["url_length"] = len(url)
    features["host_length"] = len(hostname)
    features["path_length"] = len(path)

    # Character counts
    features["count_dots"] = url.count(".")
    features["count_hyphens"] = url.count("-")
    features["count_digits"] = sum(char.isdigit() for char in url)

    # Structure
    features["num_subdomains"] = max(0, hostname.count(".") - 1)
    features["num_query_params"] = query.count("=")

    # Host / scheme checks
    features["has_ip_host"] = has_ip_address(hostname)
    features["uses_https"] = int(url.lower().startswith("https"))
    features["host_is_long"] = int(len(hostname) > 30)

    # Suspicious patterns
    features["has_at_symbol"] = int("@" in url)
    features["has_double_slash_path"] = int("//" in path)

    features["has_suspicious_word"] = int(
        any(word in url.lower() for word in SUSPICIOUS_WORDS)
    )

    # TLD & shortener checks
    tld = "." + hostname.split(".")[-1] if "." in hostname else ""
    features["has_suspicious_tld"] = int(tld in SUSPICIOUS_TLDS)
    features["is_shortener"] = int(hostname in SHORTENERS)

    # Entropy
    features["entropy"] = entropy(url)

    return features


# -------------------------
# Batch conversion
# -------------------------

def urls_to_feature_df(urls):
    rows = [extract_url_features(u) for u in urls]
    return pd.DataFrame(rows).fillna(0)
