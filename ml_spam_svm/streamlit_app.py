import streamlit as st
import pandas as pd
from data_loader import load_sms_spam
from model_utils import build_svm_pipeline, evaluate_model, save_model, load_model
from sklearn.model_selection import train_test_split
import re
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

MODEL_PATH_DEFAULT = 'models/spam_svm_model.joblib'

st.title('Spam/Ham Classifier')

st.sidebar.header('Options')
test_size = st.sidebar.slider('Test size', 0.05, 0.5, 0.2)
random_state = st.sidebar.number_input('Random state', value=42, step=1)
max_features = st.sidebar.number_input('TF-IDF max features', value=20000, step=1000)
cv = st.sidebar.number_input('Calibration CV folds (0 = no calibration)', min_value=0, max_value=10, value=3, step=1)
train_button = st.sidebar.button('Train model')

st.markdown('## Dataset')
with st.spinner('Loading dataset...'):
    try:
        df = load_sms_spam(download=True)
    except Exception as e:
        st.error(f'Failed to load dataset: {e}')
        st.stop()

st.write('Rows:', len(df))

# --- Data preview: class distribution
st.markdown('### Class distribution')
label_map = {1: 'spam', 0: 'ham'}
labels = df['label'].map(label_map)
st.bar_chart(labels.value_counts())

# --- Approximate token replacements in cleaned text
st.markdown('### Token replacements in cleaned text (approximate)')

# Define simple patterns and placeholder tokens
PATTERNS = [
    (r'https?://\S+|www\.\S+', '<URL>'),
    (r'\b[\w.-]+@[\w.-]+\.[A-Za-z]{2,}\b', '<EMAIL>'),
    (r'\$?\d+[\d,\.]*', '<NUMBER>'),
    (r'\b\d{2,4}[-/.]\d{1,2}[-/.]\d{1,2}\b', '<DATE>'),
    (r'\b\+?\d[\d\s\-()]{7,}\b', '<PHONE>'),
]

def approximate_clean_and_count(text):
    """Return (cleaned_text, replacements_count) using simple regex patterns."""
    if not isinstance(text, str):
        return '', 0
    total = 0
    cleaned = text
    for pat, token in PATTERNS:
        # count occurrences
        try:
            matches = re.findall(pat, cleaned, flags=re.IGNORECASE)
        except re.error:
            matches = []
        cnt = len(matches)
        if cnt:
            total += cnt
            cleaned = re.sub(pat, token, cleaned, flags=re.IGNORECASE)
    return cleaned, total

# Apply to a sample (avoid very slow full apply if dataset huge)
sample_df = df.copy()
res = sample_df['text'].apply(approximate_clean_and_count)
sample_df['cleaned_text'] = res.apply(lambda x: x[0])
sample_df['replacements_count'] = res.apply(lambda x: x[1])

st.write(sample_df[['text', 'cleaned_text']].head(5))

counts_series = sample_df['replacements_count'].value_counts().sort_index()
st.bar_chart(counts_series)

# --- Top tokens by class
st.markdown('---')
st.markdown('### Top tokens by class')
top_n = st.number_input('Top N tokens', min_value=5, max_value=100, value=20, step=5)
sample_limit = st.number_input('Sample size for token stats (0 = all)', min_value=0, max_value=len(sample_df), value=0, step=100)

texts = sample_df['cleaned_text'].fillna('')
if sample_limit and sample_limit > 0 and sample_limit < len(sample_df):
    texts = texts.sample(sample_limit, random_state=42)

if texts.empty:
    st.write('No text available for token analysis.')
else:
    # filter out empty texts
    texts = texts[texts.str.strip().astype(bool)]
    if texts.empty:
        st.write('No non-empty texts available for token analysis after cleaning.')
    else:
        # custom tokenizer to include tokens like <URL> and emails/numbers
        def regex_tokenizer(s):
            if not isinstance(s, str):
                return []
            return re.findall(r"[A-Za-z0-9<>@%$\.-]+", s.lower())

        try:
            vec = CountVectorizer(tokenizer=regex_tokenizer, lowercase=False)
            X = vec.fit_transform(texts)
            vocab = vec.get_feature_names_out()
            # sum columns efficiently from sparse matrix
            col_sums = X.sum(axis=0)
            counts = pd.Series(col_sums.A1, index=vocab)

            # align counts with original labels for the sampled indices
            sampled_idx = texts.index
            sampled_labels = sample_df.loc[sampled_idx, 'label']

            # Compute per-class sums by masking rows
            ham_mask = (sampled_labels == 0).values
            spam_mask = (sampled_labels == 1).values
            if ham_mask.any():
                ham_counts = X[ham_mask].sum(axis=0).A1
                ham_series = pd.Series(ham_counts, index=vocab).sort_values(ascending=False).head(top_n)
            else:
                ham_series = pd.Series(dtype=int)
            if spam_mask.any():
                spam_counts = X[spam_mask].sum(axis=0).A1
                spam_series = pd.Series(spam_counts, index=vocab).sort_values(ascending=False).head(top_n)
            else:
                spam_series = pd.Series(dtype=int)
        except ValueError as e:
            st.write(f'Token analysis skipped: {e}')
            ham_series = pd.Series(dtype=int)
            spam_series = pd.Series(dtype=int)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Ham — Top tokens')
        if not ham_series.empty:
            st.bar_chart(ham_series)
        else:
            st.write('No ham tokens in sample')
    with col2:
        st.subheader('Spam — Top tokens')
        if not spam_series.empty:
            st.bar_chart(spam_series)
        else:
            st.write('No spam tokens in sample')

st.markdown('---')

if train_button:
    X = df['text'].values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    with st.spinner('Training...'):
        model = build_svm_pipeline(max_features=max_features, cv=int(cv))
        model.fit(X_train, y_train)
    st.success('Training completed')
    metrics = evaluate_model(model, X_test, y_test)
    st.subheader('Metrics')
    st.json(metrics)
    save_model(model, MODEL_PATH_DEFAULT)
    st.write(f'Model saved to {MODEL_PATH_DEFAULT}')

st.markdown('---')


st.markdown('---')
st.header('Model Performance (Test)')
perf_model_path = st.text_input('Model path for evaluation', value=MODEL_PATH_DEFAULT, key='perf_model')
perf_button = st.button('Evaluate model on test set')

def plot_confusion_matrix(cm, labels=('ham', 'spam')):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, ylabel='True label', xlabel='Predicted label')
    fmt = 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), fmt), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    return fig

def plot_precision_recall(precision, recall, ap_score):
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple', lw=2, label=f'AP = {ap_score:.2f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower left')
    return fig

if perf_button:
    try:
        model = load_model(perf_model_path)
    except Exception as e:
        st.error(f'Failed to load model for evaluation: {e}')
        model = None
    if model is not None:
        # load data and split
        try:
            df_all = load_sms_spam(download=True)
        except Exception as e:
            st.error(f'Failed to load dataset for evaluation: {e}')
            df_all = None
        if df_all is not None:
            X = df_all['text'].values
            y = df_all['label'].values
            _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
            # predict
            try:
                y_pred = model.predict(X_test)
            except Exception as e:
                st.error(f'Prediction failed: {e}')
                y_pred = None
            if y_pred is not None:
                # compute basic classification metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                # confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                st.subheader('Confusion Matrix')
                fig_cm = plot_confusion_matrix(cm)
                st.pyplot(fig_cm)

                # scores for ROC/PR
                score = None
                try:
                    score = model.predict_proba(X_test)[:, 1]
                except Exception:
                    try:
                        score = model.decision_function(X_test)
                    except Exception:
                        score = None

                roc_auc_val = None
                ap_val = None
                if score is not None:
                    try:
                        roc_auc_val = roc_auc_score(y_test, score)
                    except Exception:
                        fpr, tpr, _ = roc_curve(y_test, score)
                        roc_auc_val = auc(fpr, tpr)
                    precision_vals, recall_vals, _ = precision_recall_curve(y_test, score)
                    ap_val = average_precision_score(y_test, score)

                # show metrics table
                metrics_table = pd.DataFrame([
                    ['accuracy', acc],
                    ['precision', prec],
                    ['recall', rec],
                    ['f1', f1],
                    ['roc_auc', roc_auc_val],
                    ['average_precision', ap_val],
                ], columns=['metric', 'value']).set_index('metric')
                st.subheader('Metrics')
                st.table(metrics_table)

                col1, col2 = st.columns(2)
                if score is not None:
                    fpr, tpr, _ = roc_curve(y_test, score)
                    roc_auc = roc_auc_val if roc_auc_val is not None else auc(fpr, tpr)
                    with col1:
                        st.subheader('ROC Curve (TPR vs FPR)')
                        fig_roc = plot_roc_curve(fpr, tpr, roc_auc)
                        st.pyplot(fig_roc)

                    with col2:
                        st.subheader('Precision-Recall Curve')
                        fig_pr = plot_precision_recall(precision_vals, recall_vals, ap_val)
                        st.pyplot(fig_pr)
                else:
                    with col1:
                        st.write('No score available for ROC/PR (model lacks probability/decision_function)')
                    with col2:
                        st.write('No score available for ROC/PR (model lacks probability/decision_function)')

st.markdown('---')
# Interactive Predict block (placed above performance)
# Ensure session state key for predict message exists
if 'predict_message' not in st.session_state:
    st.session_state['predict_message'] = ''

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button('Use spam example'):
        try:
            ex = sample_df[sample_df['label'] == 1]['text'].iloc[0]
        except Exception:
            ex = 'Free entry in 2 a wkly comp to win FA Cup. Text STOP to unsubscribe.'
        st.session_state['predict_message'] = ex
with col_b:
    if st.button('Use ham example'):
        try:
            ex = sample_df[sample_df['label'] == 0]['text'].iloc[0]
        except Exception:
            ex = 'Hey, are we still meeting tomorrow for lunch?'
        st.session_state['predict_message'] = ex

model_path = st.text_input('Model path', value=MODEL_PATH_DEFAULT, key='predict_model')
threshold = st.slider('Decision threshold (probability)', 0.0, 1.0, 0.5, 0.01)
message = st.text_area('Message to classify', value=st.session_state.get('predict_message', ''), key='predict_message', height=120)
if st.button('Predict', key='predict_button'):
    try:
        model = load_model(st.session_state.get('predict_model', MODEL_PATH_DEFAULT))
        prob = None
        raw_score = None
        # Try probability first
        try:
            prob = model.predict_proba([message])[0][1]
        except Exception:
            prob = None
        # If no prob, try decision_function and convert with sigmoid
        if prob is None:
            try:
                raw_score = model.decision_function([message])[0]
                # convert to pseudo-probability via sigmoid
                prob = 1.0 / (1.0 + np.exp(-raw_score))
            except Exception:
                prob = None
        # If still no prob, fall back to predict()
        if prob is None:
            pred = model.predict([message])[0]
            label = 'spam' if int(pred) == 1 else 'ham'
            st.write(f'Prediction: {label} (no score available)')
        else:
            label = 'spam' if float(prob) >= float(threshold) else 'ham'
            st.write(f'Prediction: {label} (probability={prob:.4f}) — threshold={threshold:.2f}')
            if raw_score is not None:
                st.write(f'Decision score: {raw_score:.4f}')
    except Exception as e:
        st.error(f'Failed to load model or predict: {e}')
