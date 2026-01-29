from src.features.url_features import extract_url_features

EXPECTED_KEYS = {
    "url_length", "host_length", "path_length",
    "count_dots", "count_hyphens", "count_digits",
    "num_subdomains", "num_query_params",
    "has_ip_host", "uses_https", "host_is_long",
    "has_at_symbol", "has_double_slash_path",
    "has_suspicious_word", "has_suspicious_tld",
    "is_shortener", "entropy"
}

def test_feature_keys_consistent():
    feats = extract_url_features("http://example.com/login")
    assert set(feats.keys()) == EXPECTED_KEYS

def test_ip_address_detected():
    feats = extract_url_features("http://192.168.1.1/login")
    assert feats["has_ip_host"] == 1

def test_shortener_detected():
    feats = extract_url_features("https://bit.ly/abc123")
    assert feats["is_shortener"] == 1

def test_entropy_non_negative():
    feats = extract_url_features("https://google.com")
    assert feats["entropy"] >= 0
