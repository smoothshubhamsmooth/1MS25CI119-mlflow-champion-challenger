# Student Name: Shubham | USN: 1MS25CI119
# Stage 3: Evaluation Dataset Preparation

import os
import pandas as pd
from datasets import load_dataset

STUDENT_NAME = "Shubham"
STUDENT_USN = "1MS25CI119"

# Fixed random seed
SEED = 42
SAMPLE_SIZE = 100

print("====================================================")
print(f"Student: {STUDENT_NAME} | USN: {STUDENT_USN}")
print("Stage 3: Preparing Shared Evaluation Dataset")
print("====================================================")

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

print("\nDownloading IMDB dataset...")

# Load the test portion of IMDB
dataset = load_dataset(
    "stanfordnlp/imdb",
    split="test"
)

# Shuffle consistently and select 100 reviews
dataset = dataset.shuffle(seed=SEED).select(
    range(SAMPLE_SIZE)
)

# Convert into a pandas DataFrame
df = pd.DataFrame({
    "text": dataset["text"],
    "label": dataset["label"]
})

# Save the evaluation dataset
df.to_csv(
    "data/eval_inputs.csv",
    index=False
)

print("\nDataset created successfully!")
print(df.head())

print("\nDataset information:")
print("Shape:", df.shape)
print(df["label"].value_counts())

print("\nSaved to data/eval_inputs.csv")
