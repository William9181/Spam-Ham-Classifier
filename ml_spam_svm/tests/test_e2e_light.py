def test_e2e_train_save_load(tmp_path):
    """Lightweight end-to-end: train a tiny SVM pipeline, save, load, predict."""
    from ml_spam_svm.model_utils import build_svm_pipeline, save_model, load_model

    # tiny synthetic dataset
    X = ["Free entry now", "Hello friend", "Win money now", "See you tomorrow"]
    y = [1, 0, 1, 0]

    pipe = build_svm_pipeline(max_features=100)
    pipe.fit(X, y)

    model_path = tmp_path / "spam_model.joblib"
    save_model(pipe, str(model_path))
    assert model_path.exists()

    loaded = load_model(str(model_path))
    preds = loaded.predict(["Free prize", "Good morning"])
    assert len(preds) == 2
    # ensure outputs are 0/1 labels
    assert all(int(p) in (0, 1) for p in preds)
