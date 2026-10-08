# Phishing Detection Analysis

Small research-oriented Python module for Assignment 3, related to the dissertation
“Comparative Analysis of Traditional Machine Learning and Deep Learning Approaches
for Phishing Detection in Remote Work Environments”.

## Purpose
The module extracts simple URL features and trains a lightweight Logistic Regression
classifier. It is an educational/research module, not a production security system.

## Technologies
Python 3.11, pandas, scikit-learn, pytest, Git, GitHub Actions.

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

## CI/CD
GitHub Actions installs dependencies and runs the automated tests on every push and pull request.

## Limitation
The included dataset is intentionally small for demonstration. A real experiment should use
a larger validated dataset with separate training and testing data.
