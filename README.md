# Champion–Challenger Model Comparison with Hugging Face and MLflow

- **Student Name:** Shubham
- **USN:** `1MS25CI119`
- **Repository Name:** `1MS25CI119-mlflow-champion-challenger`
- **Task:** Binary Sentiment Classification (`POSITIVE` / `NEGATIVE`)

---

## 1. Project Overview
This project implements an end-to-end MLOps Champion–Challenger evaluation, experiment tracking, model registration, and alias-based deployment workflow using **Hugging Face Transformers** and **MLflow**. Two pretrained sentiment classification models are evaluated on an identical 100-sample subset of the IMDb movie review test dataset (`seed=42`).

## 2. Selected Task and Models
- **Task:** Binary Sentiment Classification (`0 = NEGATIVE`, `1 = POSITIVE`)
- **Model A — Initial Champion (`champion_candidate`):** `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (66M parameters, lightweight DistilBERT architecture fine-tuned on SST-2)
- **Model B — Challenger (`challenger_candidate`):** `siebert/sentiment-roberta-large-english` (355M parameters, RoBERTa-Large architecture fine-tuned across 15 sentiment datasets)

## 3. Dataset Details
- **Source Dataset:** `stanfordnlp/imdb` (`test` split) loaded via Hugging Face `datasets`
- **Sampling Strategy:** Shuffled with fixed random seed `SEED = 42` and first `100` examples selected (`data/eval_inputs.csv`)
- **Class Distribution:** `53` Negative (`label = 0`) and `47` Positive (`label = 1`) reviews
- **Unseen Inference Data:** `8` completely new movie reviews stored in `data/new_data.csv`

## 4. Predefined Champion–Challenger Decision Rule (Defined Before Evaluation)
Before running the final evaluation (`evaluate.py`), the following quantitative promotion rule is locked:
1. **Primary Evaluation Metric:** `macro_f1` (higher is better, weighting both positive and negative classes equally).
2. **Minimum Improvement Margin:** `0.01` (`+1.0%` absolute Macro-F1 gain).
3. **Latency Guardrail:** The Challenger's average inference latency (`avg_latency_ms`) must not exceed `5.0x` the Initial Champion's average inference latency (`latency_B <= 5.0 * latency_A`).
4. **Tie-Breaker / Promotion Logic:**
   - **Case 1 (Tie within margin):** If `|macro_f1_B - macro_f1_A| < 0.01`, the two models are considered statistically equivalent on this sample, and the **faster model** (lower `avg_latency_ms`) is selected as Champion.
   - **Case 2 (Challenger promotion):** If `(macro_f1_B - macro_f1_A) >= 0.01` **AND** `avg_latency_ms_B <= 5.0 * avg_latency_ms_A`, **Model B (Challenger)** replaces Model A as the new Champion.
   - **Case 3 (Champion retention):** Otherwise (if Model B degrades quality, fails to beat the `0.01` margin, or violates the `5.0x` latency guardrail), **Model A (Initial Champion)** remains Champion.

## 5. Folder Structure
```text
1MS26CI405-mlflow-champion-challenger/
├── data/
│   ├── eval_inputs.csv
│   └── new_data.csv
├── outputs/
│   ├── confusion_matrix_champion_candidate.png
│   ├── confusion_matrix_challenger_candidate.png
│   ├── disagreements.csv
│   ├── disagreements_qualitative.json
│   ├── metrics_champion_candidate.json
│   ├── metrics_challenger_candidate.json
│   ├── metrics_summary.json
│   ├── model_comparison.png
│   ├── new_data_predictions.csv
│   ├── predictions_champion_candidate.csv
│   ├── predictions_challenger_candidate.csv
│   ├── registry_info.json
│   └── run_ids.json
├── screenshots/
│   ├── 01_model_test.png
│   ├── 02_evaluation_data.png
│   ├── mlflow_compare.png
│   ├── mlflow_registry.png
│   └── predict_from_mlflow.png
├── evaluate.py
├── predict_from_mlflow.py
├── prepare_data.py
├── register_champion.py
├── requirements.txt
├── test_models.py
├── .gitignore
└── README.md
```

## 6. Installation & Execution Instructions

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Start the MLflow Tracking & Registry Server (Keep Terminal 1 Running)
```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

### Step 3: Run the Pipeline Scripts in Order (Terminal 2)
```powershell
python test_models.py
python prepare_data.py
python evaluate.py
python register_champion.py
python predict_from_mlflow.py
```

---

## 7. Actual Evaluation Results (To Be Updated in Stage 11 After Running `evaluate.py`)
| Metric | Model A (`distilbert-base-uncased-finetuned-sst-2-english`) | Model B (`sentiment-roberta-large-english`) | Difference (`B - A`) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | `[Run evaluate.py]` | `[Run evaluate.py]` | `[Run evaluate.py]` |
| **Precision** | `[Run evaluate.py]` | `[Run evaluate.py]` | `[Run evaluate.py]` |
| **Recall** | `[Run evaluate.py]` | `[Run evaluate.py]` | `[Run evaluate.py]` |
| **F1 Score** | `[Run evaluate.py]` | `[Run evaluate.py]` | `[Run evaluate.py]` |
| **Macro-F1 (Primary)** | `[Run evaluate.py]` | `[Run evaluate.py]` | `[Run evaluate.py]` |
| **Avg Latency (ms/sample)** | `[Run evaluate.py]` | `[Run evaluate.py]` | `[Run evaluate.py]` |
| **Total Inference Time (s)** | `[Run evaluate.py]` | `[Run evaluate.py]` | `[Run evaluate.py]` |

## 8. Final Champion–Challenger Decision & Analysis
*(Will be populated with your exact 150–250 word analysis in Stage 11 once `evaluate.py` finishes running on your machine.)*

## 9. Screenshots
- **MLflow Run Comparison:** `screenshots/mlflow_compare.png`
- **MLflow Model Registry & Aliases:** `screenshots/mlflow_registry.png`
- **Inference via `@champion` Alias:** `screenshots/predict_from_mlflow.png`
