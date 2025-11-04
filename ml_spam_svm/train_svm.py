import argparse
from sklearn.model_selection import train_test_split
from data_loader import load_sms_spam
from model_utils import build_svm_pipeline, evaluate_model, save_model


def main(output, test_size, random_state, max_features, cv):
    df = load_sms_spam(download=True)
    X = df['text'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Pass cv through to the pipeline builder. Use cv=0 to skip calibration for speed.
    model = build_svm_pipeline(max_features=max_features, cv=cv)
    print("Training SVM pipeline (this may take a few minutes)...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    save_model(model, output)
    print(f"Saved model to {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an SVM spam classifier')
    parser.add_argument('--output', type=str, default='models/spam_svm_model.joblib')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--max-features', type=int, default=20000)
    parser.add_argument('--cv', type=int, default=3, help='If >0 use CalibratedClassifierCV with this #folds; use 0 to skip calibration for speed')
    parser.add_argument('--fast', action='store_true', help='Enable fast demo mode: reduces max_features and disables calibration')
    args = parser.parse_args()
    # If fast mode enabled, override some parameters for quicker runs
    if args.fast:
        print('Fast mode enabled: reducing features and skipping calibration')
        args.max_features = min(args.max_features, 1000)
        args.cv = 0
    main(args.output, args.test_size, args.random_state, args.max_features, args.cv)
