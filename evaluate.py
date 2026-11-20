# Student Name: Shubham | USN: 1MS25CI119
# Stages 6 & 7: Model Evaluation, MLflow Tracking, and Artifact Logging

import os
import json
import time
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from mlflow.models import infer_signature


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

STUDENT_NAME = "Shubham"
STUDENT_USN = "1MS25CI119"

MODEL_A = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

MODEL_B = (
    "siebert/sentiment-roberta-large-english"
)

EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"

SEED = 42
BATCH_SIZE = 8
MAX_LENGTH = 512
DEVICE = -1  # CPU

PRIMARY_METRIC = "macro_f1"
IMPROVEMENT_MARGIN = 0.01
LATENCY_GUARDRAIL = 5.0


os.makedirs("outputs", exist_ok=True)

print("====================================================")
print(f"Student: {STUDENT_NAME} | USN: {STUDENT_USN}")
print("Champion-Challenger Sentiment Evaluation with MLflow")
print("====================================================")


# ==========================================
# 2. CONNECT TO MLFLOW
# ==========================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment(EXPERIMENT_NAME)

print("Connected to MLflow at http://127.0.0.1:5000!")


# ==========================================
# 3. CUSTOM MLFLOW PYTHON MODEL
# ==========================================

class SentimentPyFunc(mlflow.pyfunc.PythonModel):

    def __init__(self, model_id):
        self.model_id = model_id

    def load_context(self, context):

        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model_id,
            tokenizer=self.model_id,
            device=-1
        )

    def predict(self, context, model_input, params=None):

        texts = model_input["text"].astype(str).tolist()

        results = self.classifier(
            texts,
            batch_size=8,
            truncation=True,
            max_length=512
        )

        predictions = []
        scores = []

        for result in results:

            label = result["label"].upper()

            if label in ["POSITIVE", "LABEL_1"]:
                prediction = "POSITIVE"

            elif label in ["NEGATIVE", "LABEL_0"]:
                prediction = "NEGATIVE"

            else:
                raise ValueError(
                    f"Unexpected sentiment label: {label}"
                )

            predictions.append(prediction)
            scores.append(result["score"])

        return pd.DataFrame({
            "text": texts,
            "prediction": predictions,
            "score": scores
        })


# ==========================================
# 4. LOAD EVALUATION DATA
# ==========================================

print("\nReading evaluation dataset...")

df = pd.read_csv("data/eval_inputs.csv")

texts = df["text"].astype(str).tolist()

actual = df["label"].astype(int).tolist()

print("Total evaluation samples:", len(df))


# ==========================================
# 5. FUNCTION TO EVALUATE ONE MODEL
# ==========================================

def evaluate_model(model_id, run_name, role):

    print("\n" + "=" * 50)
    print(f"Evaluating: {run_name} ({role})")
    print("=" * 50)

    with mlflow.start_run(run_name=run_name) as run:

        run_id = run.info.run_id

        # Log parameters
        mlflow.log_param("hf_model_id", model_id)
        mlflow.log_param("task", "sentiment-classification")
        mlflow.log_param("role", role)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("eval_samples", len(df))
        mlflow.log_param("seed", SEED)
        mlflow.log_param("device", "cpu")
        mlflow.log_param("max_length", MAX_LENGTH)

        # Log tags
        mlflow.set_tag("student_name", STUDENT_NAME)
        mlflow.set_tag("student_usn", STUDENT_USN)
        mlflow.set_tag("dataset_name", "IMDB")
        mlflow.set_tag(
            "library_versions",
            "transformers=" + __import__("transformers").__version__
            + ", torch=" + __import__("torch").__version__
            + ", mlflow=" + mlflow.__version__
        )

        # Load Hugging Face model
        print("Loading Hugging Face model...")

        classifier = pipeline(
            "sentiment-analysis",
            model=model_id,
            tokenizer=model_id,
            device=DEVICE
        )

        # Run inference and measure time
        print("Generating predictions...")

        start_time = time.perf_counter()

        results = classifier(
            texts,
            batch_size=BATCH_SIZE,
            truncation=True,
            max_length=MAX_LENGTH
        )

        total_time = time.perf_counter() - start_time

        # Convert model output into binary labels
        predictions = []
        scores = []

        for result in results:

            label = result["label"].upper()

            if label in ["POSITIVE", "LABEL_1"]:
                predictions.append(1)

            elif label in ["NEGATIVE", "LABEL_0"]:
                predictions.append(0)

            else:
                raise ValueError(
                    f"Unexpected sentiment label: {label}"
                )

            scores.append(float(result["score"]))

        # Calculate metrics
        accuracy = accuracy_score(actual, predictions)

        precision = precision_score(
            actual, predictions, zero_division=0
        )

        recall = recall_score(
            actual, predictions, zero_division=0
        )

        f1 = f1_score(
            actual, predictions, zero_division=0
        )

        macro_f1 = f1_score(
            actual, predictions,
            average="macro",
            zero_division=0
        )

        avg_latency = (total_time / len(df)) * 1000

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "macro_f1": float(macro_f1),
            "avg_latency_ms": float(avg_latency),
            "total_inference_s": float(total_time)
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Save predictions
        prediction_df = pd.DataFrame({
            "text": texts,
            "reference": actual,
            "prediction": predictions,
            "confidence": scores,
            "correct": [
                a == p for a, p in zip(actual, predictions)
            ]
        })

        prediction_file = (
            f"outputs/predictions_{role}.csv"
        )

        prediction_df.to_csv(
            prediction_file,
            index=False
        )

        # Save metrics JSON
        metrics_file = (
            f"outputs/metrics_{role}.json"
        )

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        # Confusion matrix plot
        cm = confusion_matrix(
            actual,
            predictions,
            labels=[0, 1]
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Negative", "Positive"]
        )

        display.plot(cmap="Blues")

        plt.title(f"Confusion Matrix - {role} ({STUDENT_USN})")

        plt.tight_layout()

        plot_file = (
            f"outputs/confusion_matrix_{role}.png"
        )

        plt.savefig(plot_file)
        plt.close()

        # Log artifacts
        mlflow.log_artifact(prediction_file)
        mlflow.log_artifact(metrics_file)
        mlflow.log_artifact(plot_file)
        mlflow.log_artifact("data/eval_inputs.csv")

        # Log custom MLflow model
        print("Logging model into MLflow...")

        input_example = pd.DataFrame({
            "text": ["This movie was amazing!"]
        })

        prediction_example = pd.DataFrame({
            "text": ["This movie was amazing!"],
            "prediction": ["POSITIVE"],
            "score": [0.99]
        })

        signature = infer_signature(
            input_example,
            prediction_example
        )

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SentimentPyFunc(model_id),
            input_example=input_example,
            signature=signature,
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"transformers=={__import__('transformers').__version__}",
                f"torch=={__import__('torch').__version__}",
                f"pandas=={pd.__version__}"
            ]
        )

        print("\nMetrics for", run_name)

        for key, value in metrics.items():
            print(key, ":", round(value, 4))

        return {
            "run_id": run_id,
            "metrics": metrics,
            "model_id": model_id,
            "predictions": predictions,
            "scores": scores
        }


# ==========================================
# 6. EVALUATE MODEL A AND MODEL B
# ==========================================

result_a = evaluate_model(
    MODEL_A,
    "A_sentiment_distilbert-sst2",
    "champion_candidate"
)

result_b = evaluate_model(
    MODEL_B,
    "B_sentiment_roberta-large",
    "challenger_candidate"
)


# ==========================================
# 7. CHAMPION-CHALLENGER DECISION
# ==========================================

score_a = result_a["metrics"]["macro_f1"]
score_b = result_b["metrics"]["macro_f1"]

latency_a = result_a["metrics"]["avg_latency_ms"]
latency_b = result_b["metrics"]["avg_latency_ms"]

delta = score_b - score_a

if abs(delta) < IMPROVEMENT_MARGIN:

    winner = (
        "model_A"
        if latency_a <= latency_b
        else "model_B"
    )

elif (
    delta >= IMPROVEMENT_MARGIN
    and latency_b <= LATENCY_GUARDRAIL * latency_a
):

    winner = "model_B"

else:

    winner = "model_A"


print("\n" + "=" * 50)
print(f"CHAMPION-CHALLENGER RESULT ({STUDENT_NAME} - {STUDENT_USN})")
print("=" * 50)

print("Model A Macro-F1:", round(score_a, 4))
print("Model B Macro-F1:", round(score_b, 4))
print("Difference (B - A):", round(delta, 4))
print("Model A Avg Latency (ms):", round(latency_a, 2))
print("Model B Avg Latency (ms):", round(latency_b, 2))
print("Latency Ratio (B / A):", round(latency_b / max(latency_a, 1e-9), 2))
print("Winner:", winner)


# ==========================================
# 8. SAVE METRICS SUMMARY & DISAGREEMENT EXAMPLES
# ==========================================

metrics_summary = {
    "student_name": STUDENT_NAME,
    "student_usn": STUDENT_USN,
    "primary_metric": PRIMARY_METRIC,
    "improvement_margin": IMPROVEMENT_MARGIN,
    "latency_guardrail_multiplier": LATENCY_GUARDRAIL,
    "model_A": {
        "hf_model_id": MODEL_A,
        "run_name": "A_sentiment_distilbert-sst2",
        "metrics": result_a["metrics"]
    },
    "model_B": {
        "hf_model_id": MODEL_B,
        "run_name": "B_sentiment_roberta-large",
        "metrics": result_b["metrics"]
    },
    "delta_macro_f1": float(delta),
    "latency_ratio_b_over_a": float(latency_b / max(latency_a, 1e-9)),
    "winner": winner
}

metrics_summary_file = "outputs/metrics_summary.json"
with open(metrics_summary_file, "w") as f:
    json.dump(metrics_summary, f, indent=4)

# Extract qualitative disagreement examples between Model A and Model B
label_map = {0: "NEGATIVE (0)", 1: "POSITIVE (1)"}
disagreements = []

for idx, (txt, ref, pred_a, pred_b, conf_a, conf_b) in enumerate(
    zip(
        texts,
        actual,
        result_a["predictions"],
        result_b["predictions"],
        result_a["scores"],
        result_b["scores"]
    )
):
    if pred_a != pred_b:
        if pred_b == ref and pred_a != ref:
            comment = "Model B (RoBERTa-large) correctly classified nuanced sentiment where Model A (DistilBERT) misclassified."
        elif pred_a == ref and pred_b != ref:
            comment = "Model A (DistilBERT) matched the reference label whereas Model B (RoBERTa-large) misclassified."
        else:
            comment = "Models disagreed on binary label."

        disagreements.append({
            "sample_index": int(idx),
            "review_snippet": txt[:300] + ("..." if len(txt) > 300 else ""),
            "reference_label": label_map[ref],
            "model_a_prediction": label_map[pred_a],
            "model_a_confidence": round(conf_a, 4),
            "model_b_prediction": label_map[pred_b],
            "model_b_confidence": round(conf_b, 4),
            "comment": comment
        })

disagreements_df = pd.DataFrame(disagreements)
disagreements_csv = "outputs/disagreements.csv"
disagreements_df.to_csv(disagreements_csv, index=False)

top_5_disagreements = {
    "total_disagreements_found": len(disagreements),
    "top_5_qualitative_examples": disagreements[:5]
}
disagreements_json = "outputs/disagreements_qualitative.json"
with open(disagreements_json, "w") as f:
    json.dump(top_5_disagreements, f, indent=4)

print(f"\nTotal disagreements between Model A and Model B: {len(disagreements)}")
print("Saved top qualitative disagreement examples to outputs/disagreements_qualitative.json")


# ==========================================
# 9. CREATE COMPARISON PLOT
# ==========================================

metric_names = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1"
]

values_a = [
    result_a["metrics"][m] for m in metric_names
]

values_b = [
    result_b["metrics"][m] for m in metric_names
]

x = range(len(metric_names))

plt.figure(figsize=(10, 5))

plt.bar(
    [i - 0.2 for i in x],
    values_a,
    width=0.4,
    label="Model A (DistilBERT)"
)

plt.bar(
    [i + 0.2 for i in x],
    values_b,
    width=0.4,
    label="Model B (RoBERTa-Large)"
)

plt.xticks(list(x), metric_names)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title(f"Champion vs Challenger Comparison ({STUDENT_NAME} - {STUDENT_USN})")
plt.legend()
plt.tight_layout()

comparison_plot = "outputs/model_comparison.png"

plt.savefig(comparison_plot)
plt.close()


# ==========================================
# 10. SUMMARY MLFLOW RUN
# ==========================================

with mlflow.start_run(
    run_name="compare_sentiment"
) as summary_run:

    mlflow.log_param("primary_metric", PRIMARY_METRIC)
    mlflow.log_param("margin", IMPROVEMENT_MARGIN)
    mlflow.log_param(
        "latency_guardrail",
        LATENCY_GUARDRAIL
    )

    mlflow.log_metric("delta_primary_metric", delta)
    mlflow.log_metric("model_a_macro_f1", score_a)
    mlflow.log_metric("model_b_macro_f1", score_b)
    mlflow.log_metric("model_a_avg_latency_ms", latency_a)
    mlflow.log_metric("model_b_avg_latency_ms", latency_b)
    mlflow.log_metric("total_disagreements", len(disagreements))

    mlflow.set_tag("student_name", STUDENT_NAME)
    mlflow.set_tag("student_usn", STUDENT_USN)
    mlflow.set_tag("winner", winner)

    mlflow.set_tag(
        "decision_rule",
        "Macro-F1 margin 0.01 with 5x latency guardrail"
    )

    mlflow.log_artifact(comparison_plot)
    mlflow.log_artifact(metrics_summary_file)
    mlflow.log_artifact(disagreements_csv)
    mlflow.log_artifact(disagreements_json)

    if os.path.exists("README.md"):
        mlflow.log_artifact("README.md")

    if os.path.exists("requirements.txt"):
        mlflow.log_artifact("requirements.txt")

    summary_run_id = summary_run.info.run_id


# ==========================================
# 11. SAVE RUN INFORMATION
# ==========================================

run_information = {
    "student_name": STUDENT_NAME,
    "student_usn": STUDENT_USN,
    "model_a_run_id": result_a["run_id"],
    "model_b_run_id": result_b["run_id"],
    "summary_run_id": summary_run_id,
    "winner": winner,
    "registered_model_name": REGISTERED_MODEL_NAME
}

with open("outputs/run_ids.json", "w") as f:
    json.dump(run_information, f, indent=4)


print("\nEvaluation completed successfully!")
print("Winner:", winner)
print("Run information saved to outputs/run_ids.json.")
print("Check the MLflow dashboard at http://127.0.0.1:5000!")
