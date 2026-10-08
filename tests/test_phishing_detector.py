import pytest

from src.phishing_detector import (
    EvaluationResult,
    PhishingDetector,
    cross_validate,
    extract_features,
    load_dataset,
    train_test_evaluate,
)

SAMPLE = "data/sample_data.csv"


# ---------- feature extraction ----------
def test_extract_features():
    f = extract_features("https://example.com/login")
    assert f["has_https"] == 1
    assert f["dot_count"] == 1


@pytest.mark.parametrize("url, expected", [
    ("http://192.168.1.10/login", 1),
    ("http://192.168.1.10:8080/login", 1),   # port must not hide the IP
    ("http://[2001:db8::1]/login", 1),       # IPv6
    ("http://123.example.com", 0),           # digits in a name are not an IP
    ("https://example.com", 0),
])
def test_ip_like_host(url, expected):
    assert extract_features(url)["has_ip_like_host"] == expected


def test_port_and_userinfo():
    f = extract_features("http://paypal.com@evil.test:8080/x")
    assert f["at_count"] == 1
    assert f["has_port"] == 1
    assert f["host_length"] == len("evil.test")


def test_subdomains_and_keywords():
    f = extract_features("http://paypal.com.secure-login.test/verify")
    assert f["subdomain_count"] == 2
    assert f["suspicious_keyword_count"] >= 3


@pytest.mark.parametrize("url", ["", "   ", None, "not a url", "http://", "http://host:abc/"])
def test_extract_features_never_crashes(url):
    f = extract_features(url)
    assert all(isinstance(v, (int, float)) for v in f.values())


def test_scheme_is_optional():
    assert extract_features("example.com/login")["has_https"] == 0
    assert extract_features("example.com")["host_length"] == len("example.com")


# ---------- model ----------
def test_detector_can_train_and_predict():
    urls = ["https://example.com", "https://university.edu",
            "http://secure-login-example.com/account/verify",
            "http://192.168.1.10/login"]
    labels = [0, 0, 1, 1]
    model = PhishingDetector().fit(urls, labels)
    predictions = model.predict(urls)
    assert len(predictions) == len(urls)
    assert set(predictions) <= {0, 1}


@pytest.fixture(scope="module")
def sample():
    return load_dataset(SAMPLE)


def test_model_separates_obvious_cases(sample):
    model = PhishingDetector().fit(sample["url"], sample["label"])
    assert model.predict(["http://paypal-login-verify.test/confirm"]) == [1]
    assert model.predict(["https://www.wikipedia.org"]) == [0]


def test_predict_proba_range(sample):
    model = PhishingDetector().fit(sample["url"], sample["label"])
    probs = model.predict_proba(sample["url"].head(20))
    assert len(probs) == 20 and all(0.0 <= p <= 1.0 for p in probs)


def test_feature_weights_cover_all_features(sample):
    model = PhishingDetector().fit(sample["url"], sample["label"])
    assert set(model.feature_weights()) == set(extract_features("http://a.b"))


def test_save_and_load_roundtrip(sample, tmp_path):
    model = PhishingDetector().fit(sample["url"], sample["label"])
    path = tmp_path / "model.joblib"
    model.save(path)
    restored = PhishingDetector.load(path)
    urls = sample["url"].head(30)
    assert restored.predict(urls) == model.predict(urls)


def test_training_is_deterministic(sample):
    a = PhishingDetector().fit(sample["url"], sample["label"]).predict_proba(sample["url"])
    b = PhishingDetector().fit(sample["url"], sample["label"]).predict_proba(sample["url"])
    assert a == b


# ---------- evaluation ----------
def test_holdout_evaluation_on_unseen_data(sample):
    result = train_test_evaluate(sample)
    assert isinstance(result, EvaluationResult)
    assert all(0.0 <= v <= 1.0 for v in result.as_dict().values())
    assert result.f1 > 0.8  # synthetic data is easy; guards against regressions


def test_cross_validation(sample):
    scores = cross_validate(sample, folds=5)
    assert set(scores) == {"accuracy", "precision", "recall", "f1"}
    assert scores["f1"] > 0.8


def test_evaluation_on_single_class_does_not_crash():
    urls = ["http://a.com", "http://b.com", "http://c.test/login"]
    model = PhishingDetector().fit(urls, [0, 0, 1])
    result = model.evaluate(["http://a.com"], [0])
    assert result.accuracy == 1.0


# ---------- data loading ----------
def test_dataset_loader(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_text("url,label\nhttps://example.com,0\nhttp://bad-example.com,1\n")
    data = load_dataset(str(p))
    assert list(data.columns) == ["url", "label"]
    assert len(data) == 2


def test_loader_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("link,label\nhttps://example.com,0\n")
    with pytest.raises(ValueError, match="url"):
        load_dataset(p)


def test_loader_rejects_bad_labels(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("url,label\nhttps://example.com,2\n")
    with pytest.raises(ValueError, match="label"):
        load_dataset(p)


def test_bundled_dataset_is_valid(sample):
    assert len(sample) >= 100
    assert 0.1 < sample["label"].mean() < 0.5  # imbalanced, like real traffic
