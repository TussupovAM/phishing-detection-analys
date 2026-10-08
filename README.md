# Phishing Detection Analysis

[![CI](https://github.com/YOUR_USERNAME/phishing-detection-analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/phishing-detection-analysis/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)

Small research-oriented Python module developed for **Assignment 3 (Software Development and
Integration)**. It supports the dissertation *"Comparative Analysis of Traditional Machine Learning
and Deep Learning Approaches for Phishing Detection in Remote Work Environments"*.

The module extracts lexical features from URLs and trains a lightweight, interpretable Logistic
Regression classifier. It is an educational/research module, **not** a production security system.

## Features
- `extract_features(url)` - 14 lexical features (lengths, dots, hyphens, `@`, digits, subdomains,
  suspicious keywords, HTTPS, IP host incl. IPv6, port). Robust to empty/malformed input.
- `PhishingDetector` - pipeline `DictVectorizer -> StandardScaler -> LogisticRegression`
  (`class_weight="balanced"` for imbalanced data) with `fit`, `predict`, `predict_proba`,
  `evaluate`, `feature_weights` (simple explainability), `save` / `load`.
- `train_test_evaluate` (stratified hold-out) and `cross_validate` (stratified k-fold) - metrics on
  **unseen** data: accuracy, precision, recall, F1.
- `load_dataset(path)` - CSV loader that validates columns and labels.
- Command-line interface: `python -m src.cli evaluate | predict`.

## Project structure
```
src/phishing_detector.py      # features, model, evaluation, data loading
src/cli.py                    # command-line interface
tests/                        # 30 pytest tests (features, model, evaluation, loader, CLI)
data/sample_data.csv          # SYNTHETIC demo dataset: 200 URLs, 25% phishing
scripts/make_synthetic_data.py# reproducible generator of the dataset (fixed seed)
.github/workflows/tests.yml   # CI: lint -> tests (Python 3.10-3.12, coverage) -> CLI smoke test
.github/ISSUE_TEMPLATE/       # bug / feature templates;  pull_request_template.md
requirements.txt              # runtime dependencies
requirements-dev.txt          # + pytest, pytest-cov, ruff
pytest.ini  ruff.toml         # tool configuration
```

## Quick start
```bash
git clone https://github.com/YOUR_USERNAME/phishing-detection-analysis.git
cd phishing-detection-analysis
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q --cov=src      # run tests with coverage
ruff check .             # lint
```

## Usage
```bash
python -m src.cli evaluate
python -m src.cli predict "http://paypal-login-verify.test/confirm" "https://github.com"
```
```python
from src.phishing_detector import PhishingDetector, load_dataset, train_test_evaluate

data = load_dataset("data/sample_data.csv")
print(train_test_evaluate(data))                       # metrics on a held-out 30%
model = PhishingDetector().fit(data["url"], data["label"])
print(model.predict_proba(["http://192.168.1.10:8080/login"]))   # probability of phishing
print(list(model.feature_weights().items())[:3])       # most influential features
```

## CI/CD
GitHub Actions (`.github/workflows/tests.yml`) runs on every push and pull request:
1. **lint** - `ruff check .`
2. **test** - matrix Python 3.10 / 3.11 / 3.12, `pytest` with coverage (fails below 85 %),
   coverage report uploaded as an artifact, pip dependency cache
3. **smoke-cli** - runs `python -m src.cli evaluate` on the bundled dataset

## Workflow
Work is tracked in GitHub Issues (templates provided). Changes are made in feature branches and merged
into `main` through pull requests once CI is green.

## Limitations
- The bundled dataset is **synthetic** and easy: high scores on it (F1 about 0.97-0.99) do **not**
  predict real-world performance. Use a validated real dataset (e.g. PhishTank + a benign URL list)
  for real experiments.
- Only lexical URL features are used; no page content, WHOIS, or e-mail text.
- Not a security product; do not rely on it to protect real users.

## License
MIT - see [LICENSE](LICENSE).
