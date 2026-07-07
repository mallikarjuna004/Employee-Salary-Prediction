"""
=========================================================
Employee Salary Prediction - Training Script
=========================================================

Author : Your Name
Dataset: UCI Adult Income Dataset

Description:
-------------
This script

1. Loads the Adult Income dataset
2. Cleans the dataset
3. Removes unnecessary features
4. Creates preprocessing pipeline
5. Trains Random Forest model
6. Evaluates model
7. Saves trained pipeline
8. Saves evaluation metrics

=========================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "adult",
    "adult.data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "salary_pipeline.pkl"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "metrics.json"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# Dataset Columns
# =====================================================

COLUMNS = [

    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"

]

# =====================================================
# Load Dataset
# =====================================================

print("=" * 60)
print("Employee Salary Prediction")
print("=" * 60)

df = pd.read_csv(
    DATA_PATH,
    names=COLUMNS,
    header=None,
    skipinitialspace=True
)

print("\nDataset Loaded Successfully")
print(df.shape)

# =====================================================
# Data Cleaning
# =====================================================

print("\nCleaning Dataset...")

df.replace("?", np.nan, inplace=True)

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

print("Dataset Shape :", df.shape)

# =====================================================
# Remove Unnecessary Features
# =====================================================

print("\nRemoving Unnecessary Features...")

df.drop(
    columns=[
        "fnlwgt",
        "education-num"
    ],
    inplace=True
)

# =====================================================
# Features & Target
# =====================================================

X = df.drop("income", axis=1)

y = df["income"]

# =====================================================
# Identify Feature Types
# =====================================================

categorical_features = X.select_dtypes(
    include="object"
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude="object"
).columns.tolist()

print("\nCategorical Features")

print(categorical_features)

print("\nNumeric Features")

print(numeric_features)

# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTrain Samples :", len(X_train))

print("Test Samples :", len(X_test))

# =====================================================
# Preprocessing
# =====================================================

categorical_pipeline = Pipeline(

    steps=[

        (

            "imputer",

            SimpleImputer(

                strategy="most_frequent"

            )

        ),

        (

            "encoder",

            OneHotEncoder(

                handle_unknown="ignore"

            )

        )

    ]

)

numeric_pipeline = Pipeline(

    steps=[

        (

            "imputer",

            SimpleImputer(

                strategy="median"

            )

        )

    ]

)

preprocessor = ColumnTransformer(

    transformers=[

        (

            "cat",

            categorical_pipeline,

            categorical_features

        ),

        (

            "num",

            numeric_pipeline,

            numeric_features

        )

    ]

)

# =====================================================
# Random Forest Model
# =====================================================

model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    class_weight="balanced",

    n_jobs=-1

)

pipeline = Pipeline(

    steps=[

        ("preprocessor", preprocessor),

        ("classifier", model)

    ]

)

# =====================================================
# Training
# =====================================================

print("\nTraining Model...")

pipeline.fit(

    X_train,

    y_train

)

print("Training Completed")

# =====================================================
# Prediction
# =====================================================

predictions = pipeline.predict(X_test)

# =====================================================
# Evaluation
# =====================================================

accuracy = accuracy_score(

    y_test,

    predictions

)

precision = precision_score(

    y_test,

    predictions,

    pos_label=">50K"

)

recall = recall_score(

    y_test,

    predictions,

    pos_label=">50K"

)

f1 = f1_score(

    y_test,

    predictions,

    pos_label=">50K"

)

print("\n")

print("=" * 60)

print("MODEL PERFORMANCE")

print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report\n")

print(

    classification_report(

        y_test,

        predictions

    )

)

print("\nConfusion Matrix\n")

print(

    confusion_matrix(

        y_test,

        predictions

    )

)

# =====================================================
# Save Model
# =====================================================

joblib.dump(

    pipeline,

    MODEL_PATH

)

print("\nModel Saved Successfully")

print(MODEL_PATH)

# =====================================================
# Save Metrics
# =====================================================

metrics = {

    "accuracy": round(float(accuracy), 4),

    "precision": round(float(precision), 4),

    "recall": round(float(recall), 4),

    "f1_score": round(float(f1), 4),

    "train_samples": len(X_train),

    "test_samples": len(X_test),

    "features_used": X.columns.tolist(),

    "categorical_features": categorical_features,

    "numeric_features": numeric_features,

    "target_classes": sorted(y.unique().tolist())

}

with open(

    METRICS_PATH,

    "w"

) as file:

    json.dump(

        metrics,

        file,

        indent=4

    )

print("\nMetrics Saved Successfully")

print(METRICS_PATH)

print("\n")

print("=" * 60)

print("PROJECT COMPLETED SUCCESSFULLY")

print("=" * 60)