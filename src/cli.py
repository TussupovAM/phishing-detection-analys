"""Command-line interface: ``python -m src.cli {evaluate,predict} ...``."""
from __future__ import annotations

import argparse

from .phishing_detector import PhishingDetector, cross_validate, load_dataset, train_test_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phishing-detector", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="hold-out and cross-validation metrics")
    ev.add_argument("--data", default="data/sample_data.csv")
    ev.add_argument("--folds", type=int, default=5)

    pr = sub.add_parser("predict", help="classify one or more URLs")
    pr.add_argument("urls", nargs="+")
    pr.add_argument("--data", default="data/sample_data.csv", help="training data")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data = load_dataset(args.data)
    if args.command == "evaluate":
        print(f"Samples: {len(data)}  phishing share: {data['label'].mean():.2%}")
        hold = train_test_evaluate(data)
        print("Hold-out (70/30, stratified):",
              {k: round(v, 3) for k, v in hold.as_dict().items()})
        cv = cross_validate(data, folds=args.folds)
        print(f"{args.folds}-fold cross-validation (mean):",
              {k: round(v, 3) for k, v in cv.items()})
    else:
        model = PhishingDetector().fit(data["url"], data["label"])
        for url, p in zip(args.urls, model.predict_proba(args.urls), strict=True):
            print(f"{'PHISHING' if p >= 0.5 else 'legitimate':<10} p={p:.2f}  {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
