import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Imported joblib to save the model and label encoder
import joblib


print("Loading dataset...")

#loading the dataset

df = pd.read_csv("cleaned_burnout_dataset_engineered.csv")

print("Dataset shape:", df.shape)


# Separate features and target
X = df.drop(columns=["burnout_level"])
y = df["burnout_level"]

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)


# We tried predicting 4 burnout levels but only got around 58% accuracy
# One reason is that High and Moderate overlap a lot in the data
# So we decided to simplify to two categories instead:
# At Risk = High or Severe
# Not At Risk = Low or Moderate
# This is also more useful in practice - a company just needs to know who to check on

print("Changing target to binary: At Risk / Not At Risk")

#Assigning the classes to binary labels
y_binary = y.map(lambda x: "At Risk" if x in ["High", "Severe"] else "Not At Risk")


print("Binary class distribution:")
print(y_binary.value_counts())


# Encode the binary target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_binary)

print("Target classes:", label_encoder.classes_)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

#Confirming the encoding worked and the features are there.
print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)


# We also added class_weight="balanced" to the models
# This tells the model to pay more attention to the smaller class
# Without this, the model can be lazy and just predict the majority class

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    ),
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
        "F1-score": f1
    })

    print(f"{model_name} Accuracy:", accuracy)
    print(f"{model_name} F1-score:", f1)

    print(classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_
    ))


results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-score", ascending=False)

print("\nFinal model comparison (binary):")
print(results_df)

results_df.to_csv("model_results_binary.csv", index=False)
print("Results saved as model_results_binary.csv")


# Confusion matrix for best model (Logistic Regression)
print("\nCreating confusion matrix for Logistic Regression...")

lr_model = models["Logistic Regression"]
lr_predictions = lr_model.predict(X_test)

cm = confusion_matrix(y_test, lr_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot(cmap="Greens")
plt.title("Confusion Matrix - Binary Logistic Regression")
plt.tight_layout()
plt.savefig("confusion_matrix_binary.png", dpi=300)
plt.show()

print("Saved: confusion_matrix_binary.png")


# Comparison chart: original 4-class vs new binary
print("\nCreating comparison chart...")

original_f1 = 0.5781  # from our original model_training_feature_engineering.py

# Creating the chart for an easy overview
labels = ["Original\n4-class LR", "Binary\nLogistic Regression", "Binary\nRandom Forest", "Binary\nKNN"]
f1_scores = [
    original_f1,
    results_df[results_df["Model"] == "Logistic Regression"]["F1-score"].values[0],
    results_df[results_df["Model"] == "Random Forest"]["F1-score"].values[0],
    results_df[results_df["Model"] == "KNN"]["F1-score"].values[0],
]
colors = ["#5b8dd9", "#2ecc71", "#2ecc71", "#2ecc71"]

plt.figure(figsize=(9, 5))
bars = plt.bar(labels, f1_scores, color=colors, width=0.5)
plt.axhline(y=original_f1, color="red", linestyle="--", linewidth=1.2, label=f"Original baseline ({original_f1})")
plt.ylim(0, 1.0)
plt.ylabel("F1-score")
plt.title("Before vs After: 4-Class vs Binary Classification")

# Assigning the values to the chart
for bar, val in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
             f"{val:.4f}", ha="center", fontsize=10)

plt.legend()
plt.tight_layout()
plt.savefig("improvement_comparison.png", dpi=300)
plt.show()

print("Saved: improvement_comparison.png")

# Save the trained components to files
joblib.dump(models["Logistic Regression"], "burnout_pipeline.pkl")
joblib.dump(label_encoder, "burnout_label_encoder.pkl")

print("saved: burnout_label_encoder.pkl")
print("Saved: burnout_pipeline.pkl")

print("\nDone!")
