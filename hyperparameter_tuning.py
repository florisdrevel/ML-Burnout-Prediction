import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

print("Loading dataset...")

df = pd.read_csv("cleaned_burnout_dataset_engineered.csv")

print("Dataset shape:", df.shape)


# Separate features and target
X = df.drop(columns=["burnout_level"])
y = df["burnout_level"]

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Use binary target
y_binary = y.map(lambda x: "At Risk" if x in ["High", "Severe"] else "Not At Risk")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_binary)

print("Target classes:", label_encoder.classes_)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)


# GridSearchCV tries different settings and picks the best one
# cv=3 means it splits the training data 3 times and averages the results
# scoring='f1_weighted' means we want the best F1-score, not just accuracy


# Logistic Regression: try different values of C
# C controls how strongly the model regularises (lower = stronger regularisation)
print("\nTuning Logistic Regression...")

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])

lr_params = {
    "model__C": [0.01, 0.1, 1, 10, 100]}

lr_grid = GridSearchCV(lr_pipeline, lr_params, cv=3, scoring="f1_weighted", n_jobs=-1)
lr_grid.fit(X_train, y_train)

print("Best C value:", lr_grid.best_params_)

lr_predictions = lr_grid.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_f1 = f1_score(y_test, lr_predictions, average="weighted")

print(f"Logistic Regression (tuned) Accuracy: {lr_accuracy:.4f}")
print(f"Logistic Regression (tuned) F1-score: {lr_f1:.4f}")


# KNN: try different values of n_neighbors (how many nearest neighbours to look at)
print("\nTuning KNN...")

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())])

knn_params = {
    "model__n_neighbors": [3, 5, 9, 15, 21]}

knn_grid = GridSearchCV(knn_pipeline, knn_params, cv=3, scoring="f1_weighted", n_jobs=-1)
knn_grid.fit(X_train, y_train)

print("Best n_neighbors:", knn_grid.best_params_)

knn_predictions = knn_grid.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_f1 = f1_score(y_test, knn_predictions, average="weighted")

print(f"KNN (tuned) Accuracy: {knn_accuracy:.4f}")
print(f"KNN (tuned) F1-score: {knn_f1:.4f}")


# Random Forest: try different numbers of trees and tree depth
# We tune on a sample first because 100,000 rows take too long with cross-validation
print("\nTuning Random Forest (using a sample to keep it fast)...")

from sklearn.utils import resample

X_sample, y_sample = resample(X_train, y_train, n_samples=16000, random_state=42, stratify=y_train)

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [20, None],
    "class_weight": ["balanced"]}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=3,
    scoring="f1_weighted",
    n_jobs=-1)
rf_grid.fit(X_sample, y_sample)

print("Best Random Forest settings:", rf_grid.best_params_)

# Train the best settings on the full training data
best_rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_estimators=rf_grid.best_params_["n_estimators"],
    max_depth=rf_grid.best_params_["max_depth"])
best_rf.fit(X_train, y_train)
rf_predictions = best_rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions, average="weighted")

print(f"Random Forest (tuned) Accuracy: {rf_accuracy:.4f}")
print(f"Random Forest (tuned) F1-score: {rf_f1:.4f}")


# Summary
print("Logistic Regression - Accuracy:", lr_accuracy, "F1:", lr_f1)
print("Random Forest - Accuracy:", rf_accuracy, "F1:", rf_f1)
print("KNN - Accuracy:", knn_accuracy, "F1:", knn_f1)

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "KNN"],
    "Best Params": [
        f"C={lr_grid.best_params_['model__C']}",
        f"n_estimators={rf_grid.best_params_['n_estimators']}, max_depth={rf_grid.best_params_['max_depth']}",
        f"n_neighbors={knn_grid.best_params_['model__n_neighbors']}"],
    "Accuracy": [lr_accuracy, rf_accuracy, knn_accuracy],
    "F1-score": [lr_f1, rf_f1, knn_f1]})

results.to_csv("model_results_tuned.csv", index=False)
print("\nSaved to model_results_tuned.csv")
