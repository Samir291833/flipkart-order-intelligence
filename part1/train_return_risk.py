import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.inspection import permutation_importance

# Load dataset
df = pd.read_csv("orders_dataset.csv")

# Target
X = df.drop(columns=["order_id", "returned"])
y = df["returned"]


# Feature groups
numeric_features = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]

categorical_features = [
    "product_category",
    "payment_method",
]


# Numeric preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])


# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])


# Combine preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])


# Stratified 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)


# Preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ("preprocessor", preprocessor),
])


# Fit ONLY on training data
X_train_processed = preprocessing_pipeline.fit_transform(X_train)

# Transform test data using the already-fitted preprocessing
X_test_processed = preprocessing_pipeline.transform(X_test)


print("Training rows:", X_train.shape[0])
print("Test rows:", X_test.shape[0])
print("Training processed shape:", X_train_processed.shape)
print("Test processed shape:", X_test_processed.shape)

dummy_model = DummyClassifier(
    strategy="most_frequent",
    random_state=42
)

dummy_model.fit(X_train_processed, y_train)

dummy_predictions = dummy_model.predict(X_test_processed)

dummy_accuracy = accuracy_score(y_test, dummy_predictions)
dummy_f1 = f1_score(y_test, dummy_predictions, pos_label=1)

print("\nDummyClassifier Baseline")
print("Accuracy:", round(dummy_accuracy, 4))
print("F1-score (returned=1):", round(dummy_f1, 4))

logistic_model = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000
)

logistic_model.fit(X_train_processed, y_train)

# Default threshold = 0.5
logistic_probabilities = logistic_model.predict_proba(X_test_processed)[:, 1]
logistic_predictions = (logistic_probabilities >= 0.5).astype(int)

logistic_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_f1 = f1_score(y_test, logistic_predictions, pos_label=1)
logistic_recall = recall_score(y_test, logistic_predictions, pos_label=1)
logistic_precision = precision_score(y_test, logistic_predictions, pos_label=1, zero_division=0)
logistic_roc_auc = roc_auc_score(y_test, logistic_probabilities)

print("\nLogistic Regression - Default Threshold (0.5)")
print("Accuracy:", round(logistic_accuracy, 4))
print("F1-score:", round(logistic_f1, 4))
print("Recall:", round(logistic_recall, 4))
print("Precision:", round(logistic_precision, 4))
print("ROC-AUC:", round(logistic_roc_auc, 4))


# Threshold sweep from 0.1 to 0.9 in steps of 0.01
thresholds = np.arange(0.10, 0.901, 0.01)

threshold_results = []

for threshold in thresholds:
    predictions = (logistic_probabilities >= threshold).astype(int)

    f1 = f1_score(y_test, predictions, pos_label=1)
    recall = recall_score(y_test, predictions, pos_label=1)
    precision = precision_score(y_test, predictions, pos_label=1, zero_division=0)

    threshold_results.append({
        "threshold": threshold,
        "f1": f1,
        "recall": recall,
        "precision": precision
    })


threshold_results_df = pd.DataFrame(threshold_results)

best_threshold_row = threshold_results_df.loc[
    threshold_results_df["f1"].idxmax()
]

print("\nLogistic Regression - Threshold Sweep")
print(
    threshold_results_df.to_string(
        index=False,
        formatters={
            "threshold": "{:.2f}".format,
            "f1": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "precision": "{:.4f}".format
        }
    )
)

print("\nBest Logistic Regression Threshold")
print("Threshold:", round(best_threshold_row["threshold"], 2))
print("F1-score:", round(best_threshold_row["f1"], 4))
print("Recall:", round(best_threshold_row["recall"], 4))
print("Precision:", round(best_threshold_row["precision"], 4))


rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [6, 10, None]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nRandom Forest - GridSearchCV")
print("Best parameters:", grid_search.best_params_)
print("Best cross-validated ROC-AUC:", round(grid_search.best_score_, 4))

best_rf_pipeline = grid_search.best_estimator_

rf_probabilities = best_rf_pipeline.predict_proba(X_test)[:, 1]

rf_test_roc_auc = roc_auc_score(
    y_test,
    rf_probabilities
)

print("Held-out test ROC-AUC:", round(rf_test_roc_auc, 4))

# Get the fitted preprocessing step and Random Forest
fitted_preprocessor = best_rf_pipeline.named_steps["preprocessor"]
rf_model = best_rf_pipeline.named_steps["classifier"]

# Get the transformed feature names
feature_names = fitted_preprocessor.get_feature_names_out()

# Get impurity-based feature importances
impurity_importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "impurity_importance": impurity_importances
})

feature_importance_df = feature_importance_df.sort_values(
    "impurity_importance",
    ascending=False
).reset_index(drop=True)

print("\nRandom Forest - Top 5 Impurity-Based Feature Importances")

print(
    feature_importance_df.head(5).to_string(
        index=False,
        formatters={
            "impurity_importance": "{:.6f}".format
        }
    )
)


# Permutation importance on the held-out test set
permutation_result = permutation_importance(
    best_rf_pipeline,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

permutation_df = pd.DataFrame({
    "feature": X_test.columns,
    "permutation_importance": permutation_result.importances_mean
})

# Only compare the original top-5 features from impurity importance.
#
# A transformed feature such as:
# payment_method_COD
# corresponds to the original feature:
# payment_method
#
# Therefore map the transformed names back to original feature names.
def original_feature_name(transformed_name):
    if transformed_name.startswith("numeric__"):
        return transformed_name.replace("numeric__", "")

    if transformed_name.startswith("categorical__"):
        name = transformed_name.replace("categorical__", "")

        for category_column in ["product_category", "payment_method"]:
            if name.startswith(category_column + "_"):
                return category_column

        return name

    return transformed_name


feature_importance_df["original_feature"] = (
    feature_importance_df["feature"].apply(original_feature_name)
)

top5_original_features = feature_importance_df.head(5)[
    "original_feature"
].tolist()

permutation_top5_df = permutation_df[
    permutation_df["feature"].isin(top5_original_features)
].copy()

permutation_top5_df = permutation_top5_df.sort_values(
    "permutation_importance",
    ascending=False
).reset_index(drop=True)

print("\nPermutation Importance for Original Top-5 Features")

print(
    permutation_top5_df.to_string(
        index=False,
        formatters={
            "permutation_importance": "{:.6f}".format
        }
    )
)


# Side-by-side comparison
comparison_df = feature_importance_df.head(5)[
    ["feature", "original_feature", "impurity_importance"]
].copy()

comparison_df = comparison_df.rename(
    columns={
        "feature": "transformed_feature"
    }
)

comparison_df["permutation_importance"] = comparison_df[
    "original_feature"
].map(
    permutation_df.set_index("feature")["permutation_importance"]
)

comparison_df["impurity_rank"] = range(1, 6)

comparison_df["permutation_rank"] = (
    comparison_df["permutation_importance"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype("Int64")
)

print("\nImpurity vs Permutation Importance - Top 5 Comparison")

print(
    comparison_df[
        [
            "impurity_rank",
            "transformed_feature",
            "original_feature",
            "impurity_importance",
            "permutation_importance",
            "permutation_rank"
        ]
    ].to_string(
        index=False,
        formatters={
            "impurity_importance": "{:.6f}".format,
            "permutation_importance": "{:.6f}".format
        }
    )
)


# ============================================================
# RANDOM FOREST THRESHOLD SWEEP
# ============================================================

rf_thresholds = np.arange(0.10, 0.901, 0.01)

rf_threshold_results = []

for threshold in rf_thresholds:

    predictions = (rf_probabilities >= threshold).astype(int)

    f1 = f1_score(y_test, predictions, pos_label=1)
    recall = recall_score(y_test, predictions, pos_label=1)
    precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    rf_threshold_results.append({
        "threshold": threshold,
        "f1": f1,
        "recall": recall,
        "precision": precision
    })

rf_threshold_df = pd.DataFrame(rf_threshold_results)

best_rf_threshold_row = rf_threshold_df.loc[
    rf_threshold_df["f1"].idxmax()
]

print("\nRandom Forest - Threshold Sweep")

print(
    rf_threshold_df.to_string(
        index=False,
        formatters={
            "threshold": "{:.2f}".format,
            "f1": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "precision": "{:.4f}".format
        }
    )
)

print("\nBest Random Forest Threshold")

print(
    "t*_rf:",
    round(best_rf_threshold_row["threshold"], 2)
)

print(
    "F1-score:",
    round(best_rf_threshold_row["f1"], 4)
)

print(
    "Recall:",
    round(best_rf_threshold_row["recall"], 4)
)

print(
    "Precision:",
    round(best_rf_threshold_row["precision"], 4)
)


# ============================================================
# SUBGROUP / ROOT-CAUSE ANALYSIS
# ============================================================

# Use the Random Forest's F1-maximising threshold found above.
rf_best_threshold = float(best_rf_threshold_row["threshold"])

# Generate final test predictions using t*_rf
rf_final_predictions = (
    rf_probabilities >= rf_best_threshold
).astype(int)

print("\nRandom Forest - Overall Test Performance at t*_rf")

overall_recall = recall_score(
    y_test,
    rf_final_predictions,
    pos_label=1
)

overall_precision = precision_score(
    y_test,
    rf_final_predictions,
    pos_label=1,
    zero_division=0
)

overall_f1 = f1_score(
    y_test,
    rf_final_predictions,
    pos_label=1
)

print("Threshold:", round(rf_best_threshold, 2))
print("F1-score:", round(overall_f1, 4))
print("Recall:", round(overall_recall, 4))
print("Precision:", round(overall_precision, 4))


# ------------------------------------------------------------
# Product-category subgroup analysis
# ------------------------------------------------------------

product_results = []

for category in sorted(X_test["product_category"].unique()):

    mask = X_test["product_category"] == category

    category_y_true = y_test.loc[mask]
    category_predictions = rf_final_predictions[mask.to_numpy()]

    category_recall = recall_score(
        category_y_true,
        category_predictions,
        pos_label=1,
        zero_division=0
    )

    category_precision = precision_score(
        category_y_true,
        category_predictions,
        pos_label=1,
        zero_division=0
    )

    product_results.append({
        "product_category": category,
        "n_test": int(mask.sum()),
        "recall": category_recall,
        "precision": category_precision
    })


product_subgroup_df = pd.DataFrame(product_results)

print("\nRandom Forest - Recall and Precision by Product Category")

print(
    product_subgroup_df.to_string(
        index=False,
        formatters={
            "recall": "{:.4f}".format,
            "precision": "{:.4f}".format
        }
    )
)


# ------------------------------------------------------------
# Payment-method subgroup analysis
# ------------------------------------------------------------

payment_results = []

for payment in sorted(X_test["payment_method"].unique()):

    mask = X_test["payment_method"] == payment

    payment_y_true = y_test.loc[mask]
    payment_predictions = rf_final_predictions[mask.to_numpy()]

    payment_recall = recall_score(
        payment_y_true,
        payment_predictions,
        pos_label=1,
        zero_division=0
    )

    payment_precision = precision_score(
        payment_y_true,
        payment_predictions,
        pos_label=1,
        zero_division=0
    )

    payment_results.append({
        "payment_method": payment,
        "n_test": int(mask.sum()),
        "recall": payment_recall,
        "precision": payment_precision
    })


payment_subgroup_df = pd.DataFrame(payment_results)

print("\nRandom Forest - Recall and Precision by Payment Method")

print(
    payment_subgroup_df.to_string(
        index=False,
        formatters={
            "recall": "{:.4f}".format,
            "precision": "{:.4f}".format
        }
    )
)


# ------------------------------------------------------------
# Identify weakest subgroup
# ------------------------------------------------------------

all_subgroups = []

for _, row in product_subgroup_df.iterrows():

    all_subgroups.append({
        "group_type": "product_category",
        "group": row["product_category"],
        "recall": row["recall"],
        "precision": row["precision"]
    })


for _, row in payment_subgroup_df.iterrows():

    all_subgroups.append({
        "group_type": "payment_method",
        "group": row["payment_method"],
        "recall": row["recall"],
        "precision": row["precision"]
    })


all_subgroups_df = pd.DataFrame(all_subgroups)

weakest_subgroup = all_subgroups_df.loc[
    all_subgroups_df["recall"].idxmin()
]

print("\nWeakest Subgroup by Recall")

print(
    "Group type:",
    weakest_subgroup["group_type"]
)

print(
    "Group:",
    weakest_subgroup["group"]
)

print(
    "Recall:",
    round(weakest_subgroup["recall"], 4)
)

print(
    "Precision:",
    round(weakest_subgroup["precision"], 4)
)

# ============================================================
#  SAVE FINAL RANDOM FOREST ARTIFACT
# ============================================================

import os
import joblib

os.makedirs("models", exist_ok=True)

model_path = "models/return_risk_model.pkl"

joblib.dump(
    best_rf_pipeline,
    model_path
)

print("\nSaved final Random Forest pipeline:")
print(model_path)


# Save the RF threshold separately for Part 3
threshold_path = "models/return_risk_threshold.txt"

with open(threshold_path, "w") as f:
    f.write(f"{rf_best_threshold:.4f}\n")

print("Saved RF threshold:")
print(threshold_path)


# ------------------------------------------------------------
# Verify the saved model immediately
# ------------------------------------------------------------

loaded_model = joblib.load(model_path)

loaded_probabilities = loaded_model.predict_proba(X_test)[:, 1]

print("\nSaved Model Verification")

print(
    "Loaded model type:",
    type(loaded_model).__name__
)

print(
    "Classifier type:",
    type(loaded_model.named_steps["classifier"]).__name__
)

print(
    "Maximum probability difference:",
    round(
        np.max(
            np.abs(
                loaded_probabilities - rf_probabilities
            )
        ),
        10
    )
)

print(
    "Saved threshold:",
    rf_best_threshold
)