import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


print("Loading cleaned dataset...")

df = pd.read_csv("cleaned_burnout_dataset_engineered.csv")

print("Dataset shape:", df.shape)


# Separate features and target
X = df.drop(columns=["burnout_level"])
y = df["burnout_level"]

print("X shape:", X.shape)
print("y shape:", y.shape)


# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

print("X shape after encoding:", X.shape)


# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("Target classes:")
print(label_encoder.classes_)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)


# Models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ]),

    "Random Forest": RandomForestClassifier(random_state=42),

    "XGBoost": XGBClassifier(
        eval_metric="mlogloss",
        random_state=42)
}


results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "F1-score": f1})

    print(f"{model_name} Accuracy:", accuracy)
    print(f"{model_name} F1-score:", f1)

    print(classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_))


results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-score", ascending=False)

print("\nFinal model comparison:")
print(results_df)

results_df.to_csv("model_results.csv", index=False)

print("Model results saved as model_results.csv")
