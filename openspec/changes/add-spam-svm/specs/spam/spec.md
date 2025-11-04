## ADDED Requirements

### Requirement: Dataset ingestion from Packt CSV
系統應該能夠從指定 URL 下載並解析 CSV 資料，該 CSV 無 header，格式為兩欄：label,text。標籤為 'spam' 或 'ham'。

#### Scenario: 下載並解析成功
- **WHEN** 系統從 `https://raw.githubusercontent.com/.../sms_spam_no_header.csv` 下載資料
- **THEN** 會產生一個資料表（或 DataFrame）包含欄位 `label` 與 `text`
- **AND** `label` 會被映射為二進位：spam=1, ham=0

#### Scenario: 資料不可得
- **WHEN** 資料 URL 無法連線
- **THEN** 系統回報可重試的錯誤（例如 HTTP 錯誤），且不會覆寫現有本地資料


### Requirement: SVM training pipeline
系統應該提供一個以 SVM 為核心的訓練管線，管線包含文字向量化（例如 TF-IDF）與 SVM 分類器（支援概率預測或以校準器包裝以提供概率）。

#### Scenario: 成功訓練模型
- **WHEN** 使用者執行訓練程式並提供訓練參數（test-size, random-state 等）
- **THEN** 系統會輸出訓練後的模型檔案（例如 joblib）並列出評估指標

#### Scenario: 交叉驗證選項
- **WHEN** 使用者指定交叉驗證參數
- **THEN** 系統會執行 k-fold CV 並回報平均與標準差的評估指標


### Requirement: Evaluation metrics and reporting
系統應該在評估步驟回傳下列指標：accuracy, precision, recall, f1, roc_auc（若模型提供概率時）。

#### Scenario: 評估報告
- **WHEN** 訓練完成並在測試集上評估
- **THEN** 輸出 JSON 或易讀的文字報告，包含所有指標


### Requirement: CLI inference
系統應該提供一個簡單的 CLI，能載入訓練好的模型並對單則或多則文字進行預測，輸出 label 與機率。

#### Scenario: 單則預測
- **WHEN** 使用者執行 `predict.py "some message" --model path` 
- **THEN** 系統輸出 `spam` 或 `ham` 與對應機率


### Requirement: Reproducibility
系統應該記錄隨機種子、依賴版本（requirements.txt）與資料來源，並在 README 中描述如何重現訓練流程。

#### Scenario: 可重現訓練
- **WHEN** 使用者依 README 的步驟與相同參數執行訓練
- **THEN** 得到可比的訓練與評估輸出（差異在可接受範圍）
