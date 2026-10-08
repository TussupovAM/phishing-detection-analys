"""Generate the SYNTHETIC demonstration dataset ``data/sample_data.csv``.

The URLs are artificial: legitimate ones combine well-known domains with typical
paths, phishing ones imitate common lexical tricks (brand + keyword hyphenation,
IP hosts, '@' tricks, deep subdomains) on reserved ``.test`` / ``.example`` domains.
It is NOT real-world data; replace it with a validated dataset for real experiments.
"""
import csv
import random
import sys
from pathlib import Path

rng = random.Random(42)
LEGIT_DOMAINS = [
    "github.com", "wikipedia.org", "python.org", "openai.com", "university.edu",
    "nytimes.com", "bbc.co.uk", "stackoverflow.com", "arxiv.org", "mozilla.org",
    "kernel.org", "nature.com", "ieee.org", "acm.org", "coursera.org",
    "microsoft.com", "apple.com", "google.com", "gov.kz", "nu.edu.kz",
    "scikit-learn.org", "pandas.pydata.org", "docs.python.org", "my-company.com",
]
LEGIT_PATHS = ["", "", "/", "/about", "/docs/index.html", "/blog/2024/new-release",
               "/search?q=python", "/products/item-123", "/login", "/account/settings",
               "/support/contact-us", "/en/news/article-2026-01-15"]
BRANDS = ["paypal", "microsoft", "apple", "amazon", "netflix", "dropbox", "bank", "office365"]
KEYWORDS = ["login", "verify", "secure", "account", "update", "password", "confirm", "support"]
TLDS = ["test", "example", "invalid"]


def legit():
    scheme = "https" if rng.random() < 0.9 else "http"
    www = "www." if rng.random() < 0.4 else ""
    return f"{scheme}://{www}{rng.choice(LEGIT_DOMAINS)}{rng.choice(LEGIT_PATHS)}"


def phish():
    scheme = "http" if rng.random() < 0.7 else "https"
    brand, kw, tld = rng.choice(BRANDS), rng.choice(KEYWORDS), rng.choice(TLDS)
    path = rng.choice(["/login", "/verify/account", "/confirm?id=" + str(rng.randint(1000, 99999)),
                       "/update/password", "", "/secure/signin.php"])
    kind = rng.choice(["hyphen", "hyphen", "ip", "subdomain", "at", "short"])
    if kind == "hyphen":
        return f"{scheme}://{brand}-{kw}-{rng.choice(KEYWORDS)}.{tld}{path}"
    if kind == "ip":
        ip = ".".join(str(rng.randint(1, 254)) for _ in range(4))
        port = rng.choice(["", "", ":8080", ":8443"])
        return f"http://{ip}{port}{path or '/login'}"
    if kind == "subdomain":
        return f"{scheme}://{brand}.com.{kw}.{rng.choice(KEYWORDS)}-portal.{tld}{path}"
    if kind == "at":
        return f"{scheme}://{brand}.com@{kw}-{rng.randint(10, 999)}.{tld}{path}"
    return f"{scheme}://{brand}{kw}.{tld}{path}"


def main(out="data/sample_data.csv", n_legit=150, n_phish=50):
    rows = [(legit(), 0) for _ in range(n_legit)] + [(phish(), 1) for _ in range(n_phish)]
    rng.shuffle(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["url", "label"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
