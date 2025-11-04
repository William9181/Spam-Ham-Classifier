## Why
針對課程作業，我們需要一個可重現且簡潔的 spam 分類基線實作。選擇 SVM 作為基礎分類器可以在小至中型文本資料上提供穩定且可解釋的結果。資料來源為公開 CSV（Packt 範例）：
https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

## What Changes
- 新增一個 spam classification 能力（基於 SVM 的機器學習管線）。
- 新增資料載入與預處理規格，明確指定 CSV 欄位與預處理步驟（tokenization、TF-IDF 等）。
- 新增訓練/評估流程（train、test split、交叉驗證選項）。
- 提供模型儲存格式與 CLI 推論接口（載入模型並對單則或多則訊息進行預測）。
- 提供可重現性說明（隨機種子、環境、requirements.txt）。

## Impact
- Affected specs: 新增 `specs/spam/spec.md`（ADDED Requirements）
- Affected code: 新增 `ml_spam/`（或在現有 ML 目錄下新增模組）包含資料載入、訓練腳本、推論 CLI、測試與 requirements。 
- 破壞性：**非破壞性**；為新增功能，不會直接改變既有 API，但後續若想整合需與現有系統協調。

## Data Source
Dataset URL:
https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

## Acceptance Criteria
- 能夠從上述 URL 下載並解析資料（CSV，無 header，預期兩欄：label,text）。
- 訓練流程能輸出 metrics：accuracy, precision, recall, f1, roc_auc（如適用）。
- CLI 能對單則訊息輸出 label 與置信度。
- 提供可執行的 `requirements.txt` 與 README（包含 PowerShell 執行範例）。