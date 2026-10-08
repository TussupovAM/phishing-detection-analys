from src.phishing_detector import PhishingDetector, extract_features, load_dataset

def test_extract_features():
    f = extract_features("https://example.com/login")
    assert f["has_https"] == 1
    assert f["dot_count"] == 1

def test_detector_can_train_and_predict():
    urls = ["https://example.com", "https://university.edu",
            "http://secure-login-example.com/account/verify",
            "http://192.168.1.10/login"]
    labels = [0, 0, 1, 1]
    model = PhishingDetector().fit(urls, labels)
    predictions = model.predict(urls)
    assert len(predictions) == len(urls)
    assert set(predictions) <= {0, 1}

def test_dataset_loader(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_text("url,label\nhttps://example.com,0\nhttp://bad-example.com,1\n")
    data = load_dataset(str(p))
    assert list(data.columns) == ["url", "label"]
    assert len(data) == 2
