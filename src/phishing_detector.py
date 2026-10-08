"""Lightweight lexical URL phishing detector (educational/research module)."""
from __future__ import annotations

import ipaddress
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SUSPICIOUS_KEYWORDS = (
    "login", "verify", "secure", "account", "update", "password",
    "confirm", "signin", "bank", "support",
)


def _is_ip(host: str) -> bool:
    """Return True if ``host`` is a valid IPv4/IPv6 address."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def extract_features(url: str) -> dict[str, float]:
    """Convert a URL into a dictionary of numeric lexical features."""
    url = (url or "").strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    try:
        port = parsed.port
    except ValueError:  # malformed port, e.g. "host:abc"
        port = None
    host = (parsed.hostname or "").lower()  # excludes port and user@ part
    path = parsed.path
    is_ip = _is_ip(host)
    labels = [p for p in host.split(".") if p]
    lowered = url.lower()
    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "slash_count": url.count("/"),
        "at_count": url.count("@"),
        "question_count": url.count("?"),
        "digit_count": sum(ch.isdigit() for ch in url),
        "subdomain_count": 0 if is_ip else max(len(labels) - 2, 0),
        "suspicious_keyword_count": sum(lowered.count(k) for k in SUSPICIOUS_KEYWORDS),
        "has_https": int(parsed.scheme.lower() == "https"),
        "has_ip_like_host": int(is_ip),
        "has_port": int(port is not None),
    }


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _metrics(labels, pred) -> EvaluationResult:
    return EvaluationResult(
        accuracy_score(labels, pred),
        precision_score(labels, pred, zero_division=0),
        recall_score(labels, pred, zero_division=0),
        f1_score(labels, pred, zero_division=0),
    )


class PhishingDetector:
    """URL classifier: DictVectorizer -> StandardScaler -> Logistic Regression."""

    def __init__(self, random_state: int = 42):
        self.model = Pipeline([
            ("vectorizer", DictVectorizer(sparse=False)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=random_state)),
        ])

    @staticmethod
    def _features(urls):
        return [extract_features(u) for u in urls]

    def fit(self, urls, labels):
        self.model.fit(self._features(urls), list(labels))
        return self

    def predict(self, urls):
        return self.model.predict(self._features(urls)).tolist()

    def predict_proba(self, urls):
        """Probability of the phishing class (label 1) for each URL."""
        return self.model.predict_proba(self._features(urls))[:, 1].tolist()

    def evaluate(self, urls, labels) -> EvaluationResult:
        return _metrics(list(labels), self.predict(urls))

    def feature_weights(self) -> dict[str, float]:
        """Standardised logistic-regression coefficients (simple explainability)."""
        names = self.model.named_steps["vectorizer"].get_feature_names_out()
        coefs = self.model.named_steps["classifier"].coef_[0]
        return dict(sorted(zip(names, coefs, strict=True), key=lambda kv: -abs(kv[1])))

    def save(self, path) -> None:
        joblib.dump(self.model, Path(path))

    @classmethod
    def load(cls, path) -> PhishingDetector:
        detector = cls()
        detector.model = joblib.load(Path(path))
        return detector


def load_dataset(path) -> pd.DataFrame:
    """Read a CSV with columns ``url`` and ``label`` (0 = legitimate, 1 = phishing)."""
    data = pd.read_csv(path)
    missing = {"url", "label"} - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not set(data["label"].unique()) <= {0, 1}:
        raise ValueError("Column 'label' must contain only 0 (legitimate) and 1 (phishing)")
    return data


def train_test_evaluate(data: pd.DataFrame, test_size: float = 0.3,
                        random_state: int = 42) -> EvaluationResult:
    """Stratified hold-out evaluation: train on one part, measure on unseen URLs."""
    x_train, x_test, y_train, y_test = train_test_split(
        data["url"], data["label"], test_size=test_size,
        stratify=data["label"], random_state=random_state)
    return PhishingDetector(random_state).fit(x_train, y_train).evaluate(x_test, y_test)


def cross_validate(data: pd.DataFrame, folds: int = 5,
                   random_state: int = 42) -> dict[str, float]:
    """Mean metrics over stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    results = []
    for train_idx, test_idx in skf.split(data["url"], data["label"]):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        model = PhishingDetector(random_state).fit(train["url"], train["label"])
        results.append(model.evaluate(test["url"], test["label"]).as_dict())
    return pd.DataFrame(results).mean().to_dict()
