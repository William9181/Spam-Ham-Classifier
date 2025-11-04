import argparse
from model_utils import load_model


def predict_text(model_path, text):
    model = load_model(model_path)
    pred = model.predict([text])[0]
    prob = None
    try:
        prob = model.predict_proba([text])[0][1]
    except Exception:
        pass
    label = 'spam' if int(pred) == 1 else 'ham'
    return label, prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict spam/ham for a given message using SVM model')
    parser.add_argument('text', type=str, help='The message text to classify')
    parser.add_argument('--model', type=str, default='models/spam_svm_model.joblib', help='Path to trained model')
    args = parser.parse_args()
    label, prob = predict_text(args.model, args.text)
    if prob is not None:
        print(f"Prediction: {label} (probability={prob:.4f})")
    else:
        print(f"Prediction: {label}")
