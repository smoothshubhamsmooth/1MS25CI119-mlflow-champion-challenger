# Student Name: Shubham | USN: 1MS25CI119
# Stage 9: MLflow Model Registry and Alias Assignment

import json
import mlflow

from mlflow import MlflowClient

STUDENT_NAME = "Shubham"
STUDENT_USN = "1MS25CI119"

# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

REGISTERED_MODEL_NAME = "sentiment-model"

print("====================================================")
print(f"Student: {STUDENT_NAME} | USN: {STUDENT_USN}")
print("Stage 9: Registering Models & Assigning Aliases")
print("====================================================")

# Read the evaluation run IDs
with open("outputs/run_ids.json", "r") as f:
    run_info = json.load(f)


run_a = run_info["model_a_run_id"]
run_b = run_info["model_b_run_id"]

winner = run_info["winner"]


# Register Model A
print("\nRegistering Model A...")

version_a = mlflow.register_model(
    model_uri=f"runs:/{run_a}/model",
    name=REGISTERED_MODEL_NAME,
    tags={
        "student_name": STUDENT_NAME,
        "student_usn": STUDENT_USN,
        "candidate_role": "Model_A_DistilBERT"
    }
)


# Register Model B
print("Registering Model B...")

version_b = mlflow.register_model(
    model_uri=f"runs:/{run_b}/model",
    name=REGISTERED_MODEL_NAME,
    tags={
        "student_name": STUDENT_NAME,
        "student_usn": STUDENT_USN,
        "candidate_role": "Model_B_RoBERTa_Large"
    }
)


# Assign aliases based on the actual winner
if winner == "model_A":

    champion_version = version_a.version
    challenger_version = version_b.version

else:

    champion_version = version_b.version
    challenger_version = version_a.version


client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="champion",
    version=champion_version
)

client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="challenger",
    version=challenger_version
)


# Print the results
print("\nRegistration completed!")
print("Student:", STUDENT_NAME, "| USN:", STUDENT_USN)
print("Registered model:", REGISTERED_MODEL_NAME)
print("Winner from evaluation:", winner)
print("Champion version (@champion):", champion_version)
print("Challenger version (@challenger):", challenger_version)


# Save registration information
registration_info = {
    "student_name": STUDENT_NAME,
    "student_usn": STUDENT_USN,
    "registered_model_name": REGISTERED_MODEL_NAME,
    "winner": winner,
    "champion_version": champion_version,
    "challenger_version": challenger_version
}

with open("outputs/registry_info.json", "w") as f:
    json.dump(registration_info, f, indent=4)

print("Saved registry details to outputs/registry_info.json")
