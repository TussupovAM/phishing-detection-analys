from src.cli import main


def test_cli_evaluate(capsys):
    assert main(["evaluate", "--folds", "3"]) == 0
    out = capsys.readouterr().out
    assert "Hold-out" in out and "cross-validation" in out


def test_cli_predict(capsys):
    assert main(["predict", "http://paypal-login-verify.test/x", "https://github.com"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("PHISHING")
    assert lines[1].startswith("legitimate")
