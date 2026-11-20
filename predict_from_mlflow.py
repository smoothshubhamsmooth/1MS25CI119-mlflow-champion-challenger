# Student Name: Shubham | USN: 1MS25CI119
# Stage 10: Champion Model Inference via MLflow Alias (No Direct Hugging Face Calls)

import mlflow
import pandas as pd

from mlflow import MlflowClient

STUDENT_NAME = "Shubham"
STUDENT_USN = "1MS25CI119"

# ==========================================
# 1. CONNECT TO MLFLOW
# ==========================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

client = MlflowClient()

MODEL_NAME = "sentiment-model"

MODEL_URI = f"models:/{MODEL_NAME}@champion"


# ==========================================
# 2. FIND THE CURRENT CHAMPION VERSION
# ==========================================

model_version = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion"
)

version_number = model_version.version

run_id = model_version.run_id

# Read Hugging Face model ID from MLflow parameters
run = client.get_run(run_id)

hf_model_id = run.data.params["hf_model_id"]


print("\n====================================")
print("CHAMPION MODEL INFORMATION")
print("====================================")
print("Student Name:", STUDENT_NAME)
print("Student USN:", STUDENT_USN)
print("Registered model:", MODEL_NAME)
print("Model alias URI:", MODEL_URI)
print("Model version:", version_number)
print("Hugging Face model:", hf_model_id)


# ==========================================
# 3. LOAD THE MODEL THROUGH MLFLOW
# ==========================================

print("\nLoading Champion through MLflow...")

champion_model = mlflow.pyfunc.load_model(
    MODEL_URI
)

print("Model loaded successfully!")


# ==========================================
# 4. LOAD NEW DATA
# ==========================================

new_data = pd.read_csv(
    "data/new_data.csv"
)

print("\nNew examples:", len(new_data))


# ==========================================
# 5. GENERATE PREDICTIONS
# ==========================================

predictions = champion_model.predict(
    new_data[["text"]]
)


# ==========================================
# 6. SAVE RESULTS
# ==========================================

output = new_data.copy()

output["prediction"] = predictions["prediction"].values

output["confidence"] = predictions["score"].values

output["model_version"] = version_number

output["hf_model_id"] = hf_model_id

output["student_usn"] = STUDENT_USN

output.to_csv(
    "outputs/new_data_predictions.csv",
    index=False
)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n====================================")
print(f"NEW DATA PREDICTIONS ({STUDENT_NAME} | {STUDENT_USN})")
print("====================================")

print(
    output[
        ["text", "prediction", "confidence"]
    ].to_string(index=False)
)

print("\nPredictions saved successfully!")
print("File: outputs/new_data_predictions.csv")
