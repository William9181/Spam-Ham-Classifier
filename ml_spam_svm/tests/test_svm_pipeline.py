def test_build_svm_pipeline_smoke():
    from ml_spam_svm.model_utils import build_svm_pipeline
    pipe = build_svm_pipeline(max_features=1000)
    assert hasattr(pipe, 'fit')
    assert hasattr(pipe, 'predict')


def test_train_predict_small_sample():
    # Use a tiny synthetic dataset to test train/predict flow
    from ml_spam_svm.model_utils import build_svm_pipeline
    X = ["Free entry now", "Hello friend", "Win money now", "See you tomorrow"]
    y = [1, 0, 1, 0]
    pipe = build_svm_pipeline(max_features=100)
    pipe.fit(X, y)
    preds = pipe.predict(["Free prize", "Good morning"])
    assert len(preds) == 2
