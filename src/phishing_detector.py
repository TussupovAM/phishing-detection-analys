from dataclasses import dataclass
from urllib.parse import urlparse
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

def extract_features(url):
    parsed = urlparse(url if "://" in url else "http://" + url)
    host, path = parsed.netloc, parsed.path
    return {
        "url_length": len(url), "host_length": len(host), "path_length": len(path),
        "dot_count": url.count("."), "hyphen_count": url.count("-"),
        "slash_count": url.count("/"), "at_count": url.count("@"),
        "question_count": url.count("?"),
        "has_https": int(parsed.scheme.lower() == "https"),
        "has_ip_like_host": int(host.replace(".", "").isdigit()),
    }

@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float

class PhishingDetector:
    def __init__(self):
        self.model = Pipeline([
            ("vectorizer", DictVectorizer(sparse=False)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])

    def fit(self, urls, labels):
        self.model.fit([extract_features(u) for u in urls], labels)
        return self

    def predict(self, urls):
        return self.model.predict([extract_features(u) for u in urls]).tolist()

    def evaluate(self, urls, labels):
        pred = self.predict(urls)
        return EvaluationResult(
            accuracy_score(labels, pred),
            precision_score(labels, pred, zero_division=0),
            recall_score(labels, pred, zero_division=0),
            f1_score(labels, pred, zero_division=0),
        )

def load_dataset(path):
    data = pd.read_csv(path)
    missing = {"url", "label"} - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return data
