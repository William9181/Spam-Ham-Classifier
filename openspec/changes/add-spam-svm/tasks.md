## 1. Implementation
- [ ] 1.1 建立資料下載與解析程式（`openspec/changes/add-spam-svm/data_loader.py` 或 `ml_spam_svm/data_loader.py`）
  - 驗收：能從
    `https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv`
    下載並解析成 DataFrame，包含 `label`、`text` 並將 `label` 映射為 spam=1, ham=0。
- [ ] 1.2 建立 SVM 訓練腳本（`ml_spam_svm/train_svm.py`）
  - 要點：使用 `TfidfVectorizer` + `LinearSVC` 或 `SVC` + `CalibratedClassifierCV` 以取得機率輸出
  - 支援參數：`--test-size`、`--random-state`、`--cv`（選用）
  - 驗收：執行後會在指定路徑輸出模型檔（joblib）並列出 metrics
- [ ] 1.3 模型儲存與載入工具（`ml_spam_svm/model_utils.py`）
  - 驗收：有 `save_model(path, model)` 與 `load_model(path)`，模型格式為 joblib
- [ ] 1.4 建立 CLI 推論程式（`ml_spam_svm/predict_svm.py`）
  - 驗收：能載入模型並對單則文字輸出 `spam/ham` 與機率（若有）
- [ ] 1.5 更新 `.gitignore` 以排除模型二進位檔（`models/`）

## 2. Tests
- [ ] 2.1 新增單元測試（`tests/test_spam_svm.py`）
  - 包含：資料載入（在離線情況下可跳過）、pipeline 建構、簡短訓練與推論 smoke test
  - 驗收：`pytest` 本地執行通過（若資料不存在，測試有明確跳過條件）
- [ ] 2.2 加入小型端到端測試（在 CI 中執行完整 train -> predict，使用小子集或 mock）

## 3. Docs / Reproducibility
- [ ] 3.1 新增 `ml_spam_svm/requirements.txt`（列出 scikit-learn, pandas, joblib, requests 等）
- [ ] 3.2 更新 `README.md`（下載資料、訓練、測試、推論的步驟，包含 PowerShell 範例）
- [ ] 3.3 記錄版本與隨機種子（在 README 或訓練輸出中顯示）

## 4. CI / Validation
- [ ] 4.1 建立 GitHub Actions workflow（`.github/workflows/ci-spam-svm.yml`）
  - 在 PR 上執行：`pip install -r requirements.txt`、`pytest`、`openspec validate add-spam-svm --strict`（如可用）
  - 驗收：PR 檢查（CI）能顯示測試與 openspec 檢驗結果
- [ ] 4.2 (選項) 在 PR 中執行黑盒/效能檢測（若需要）

## 5. Release / Archive
- [ ] 5.1 完成所有任務並在 `openspec/changes/add-spam-svm/tasks.md` 中把核取方塊打勾（`- [x]`）
- [ ] 5.2 建立 PR，等待審核；合併後執行 `openspec archive add-spam-svm`（或手動移動到 archive）

---

Notes:
- 每項 Implementation 任務請盡量拆成小步（30–120 分鐘可完成）以便逐步提交與審核。
- 若團隊有特定 CI 限制（例如不能在 CI 下載外部資料），請在 2.2 中改為使用快取或 mock dataset。
- 若你希望我代為實作任一小項（例如 1.1 或 1.2），請告訴我要先做哪一項，我會把該項目標為 in-progress 並提交實作檔案。
