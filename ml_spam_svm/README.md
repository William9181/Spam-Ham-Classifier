# Spam/Ham Classifier (SVM baseline)

本目錄包含一個使用 TF‑IDF + LinearSVC 的簡單可重現垃圾郵件 (spam) 判別基線，並提供 CLI 與 Streamlit 互動介面以供訓練、推論與模型檢視。

## 資料來源
- 使用公開的 SMS spam 資料集（Packt 提供的 CSV，無 header，兩欄 — label, text）。程式會在需要時自動從原始 URL 下載並儲存在記憶體中。
- 在程式中載入資料的函式：`ml_spam_svm/data_loader.py::load_sms_spam()`。

## 模型與訓練流程
- 特徵：TF‑IDF（預設 ngram=(1,2)，可透過 CLI 調整 `--max-features` 開關）。
- 分類器：`sklearn.svm.LinearSVC`。
- 機率輸出：LinearSVC 本身不提供校準機率；若需要可靠的概率（`predict_proba`），訓練時可啟用校準（CalibratedClassifierCV）透過 `--cv` 設定折數（>0）。
- 快速模式：`--fast` 可關閉校準與降低特徵數以加快測試／展示。
- 模型儲存：預設路徑 `models/spam_svm_model.joblib`（可透過 `--output` 指定），使用 joblib 序列化。

核心訓練程式：`ml_spam_svm/train_svm.py`

訓練範例（PowerShell）：

```powershell
# 建議先安裝 requirements
pip install -r ml_spam_svm/requirements.txt

# 訓練並使用 3 折校準（會產生可用於 predict_proba 的模型）
python ml_spam_svm\train_svm.py --output models/spam_svm_model.joblib --cv 3

# 快速模式（較少特徵、無校準，適合本機快速示範）
python ml_spam_svm\train_svm.py --output models/spam_svm_model.joblib --fast
```

訓練完成後，模型會存到指定路徑。若使用 `--cv > 0`，得到的模型會支援 `predict_proba`（或使用 `CalibratedClassifierCV` 提供的概率），否則只能使用 `decision_function` 或直接 `predict`。

## 推論（CLI）
- 檔案：`ml_spam_svm/predict_svm.py`
- 範例：

```powershell
python ml_spam_svm\predict_svm.py --model models/spam_svm_model.joblib "Free entry in 2 a wkly comp to win FA Cup. Text STOP to unsubscribe"
```

當模型不提供 `predict_proba` 時，CLI/Streamlit 會嘗試呼叫 `decision_function` 並以 sigmoid 將分數轉為 0–1 的「pseudo-probability」，以便支援閾值 (threshold) 控制。

## 互動介面（Streamlit）
- 檔案：`ml_spam_svm/streamlit_app.py`
- 可以即時檢視資料分布、token 統計、Top tokens、訓練按鈕與模型效能（混淆矩陣、ROC、Precision‑Recall），以及單筆文字的互動推論。
- 啟動方式：

```powershell
streamlit run ml_spam_svm\streamlit_app.py
```

主要功能亮點：
- 範例按鈕：快速把測試範例填入預測文本欄位（spam / ham）。
- 閾值滑桿：可動態調整預測的決策閾值（0.0–1.0），Streamlit 會用 `predict_proba`（若可用）或 `sigmoid(decision_function)` 來做機率估計，並根據閾值回傳 `spam` / `ham`。

## 依賴（requirements）
- 請參考 `ml_spam_svm/requirements.txt`。
- 大致包含：pandas, scikit-learn, joblib, requests, streamlit, matplotlib, numpy。

安裝：

```powershell
pip install -r ml_spam_svm/requirements.txt
```

如果你使用虛擬環境，請先啟用該環境再安裝。

## 測試
- 使用 pytest 執行專案內的單元測試與輕量 e2e 測試：

```powershell
pytest -q
```

## 常見操作／提示
- 若想要在 Streamlit 中查看校準後的真實機率，請以 `--cv` 參數在訓練時啟用校準（例如 `--cv 3`）。
- 若模型沒有 `predict_proba`，Streamlit 與 CLI 仍會嘗試用 `decision_function` 並用 sigmoid 轉換為 (0,1)；但請注意此值不是校準機率，僅作閾值參考。
- 若資料/模型檔過大，建議把 `models/` 加到 `.gitignore`（本專案預設已忽略 models/）。

## 進階／延伸
- 若要取得更好的概率估計，可在訓練階段採用 cross‑validation 校準或改用能本身輸出概率的分類器（例如 `LogisticRegression`）。
- 若需要更豐富的互動圖表，可考慮引入 Plotly 或 Altair（需在 `requirements.txt` 中新增相應依賴並更新 CI）。

---

如果你希望我把 README 翻成英文或加上更詳細的操作範例（例如 CI badge、打包、或在 Docker 中執行），我可以接著幫你補完。
![CI](https://github.com/<your-user>/<your-repo>/actions/workflows/ci.yml/badge.svg)

SVM spam classifier baseline

Overview

This implementation trains a Support Vector Machine (SVM) classifier on the provided CSV dataset. It includes data loading, training with probability calibration, evaluation, model saving, and a CLI for inference.

Dataset

URL:
https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

Quick start (PowerShell)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r ml_spam_svm\requirements.txt
python ml_spam_svm\train_svm.py --output models/spam_svm_model.joblib --test-size 0.2 --random-state 42
python ml_spam_svm\predict_svm.py "Free entry in 2 a wkly comp to win FA Cup..." --model models/spam_svm_model.joblib
```

Notes
- The dataset will be downloaded into `ml_spam_svm/data/`.
- The model is saved as a joblib file. Add `models/` to `.gitignore` to avoid committing binaries.

Streamlit app

You can run a Streamlit app that lets you explore the dataset, train the SVM interactively, and make predictions.

```powershell
streamlit run ml_spam_svm\streamlit_app.py
```

Pushing to GitHub

To push this repository to a new GitHub repo (example):

```powershell
git init
git add .
git commit -m "Init HW3 with spam SVM"
git branch -M main
# create repo on GitHub (use gh CLI or web) and then:
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

Fast train/demo mode

If you want a quicker training run for demos or interactive testing (disables calibration and lowers TF-IDF features), use the `--fast` flag:

```powershell
python ml_spam_svm\train_svm.py --output models/spam_svm_model.joblib --fast
```

This is useful for Streamlit interactive sessions or when you want a fast feedback loop.
