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

# To save the model and label encoder
import joblib

# VotingClassifier to let models vote
from sklearn.ensemble import VotingClassifier

print("Loading dataset...")

# Loading the dataset
df = pd.read_csv("company_data.csv")

print("Dataset shape:", df.shape)

# We don't want names or IDs messing up the pattern finding, and burnout_level is what we predict
columns_to_drop = ["burnout_level", "name", "employee_id"]

X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Setting target class
y = df["burnout_level"]

# Encode categorical features into 0s and 1s
X = pd.get_dummies(X, drop_first=True)

# We tried predicting 4 burnout levels but only got around 58% accuracy
# One reason is that High and Moderate overlap a lot in the data
# So we decided to simplify to two categories instead:
# At Risk = High or Severe
# Not At Risk = Low or Moderate
# This is also more useful in practice - a company just needs to know who to check on

print("Changing target to binary: At Risk / Not At Risk")

# Assigning the classes to binary labels using a lambda function
y_binary = y.map(lambda x: "At Risk" if x in ["High", "Severe"] else "Not At Risk")

# Printing out the new counts to see if the two groups are balanced or uneven
print("Binary class distribution:")
print(y_binary.value_counts())

# Encode the binary text labels to 0 and 1
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_binary)

# Double-checking which number represents which text label
print("Target classes:", label_encoder.classes_)

# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Confirming the encoding worked and the features are there.
print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)

# We also added class_weight="balanced" to the models
# This tells the model to pay more attention to the smaller class
# Without this, the model can be lazy and just predict the majority class

# The different models and their pipelines
models = {
    # Logistic Regression needs data on the same scale, so we bundle a scaler with it
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"))
    ]),

    # KNN looks at close neighbors. We use distance weights so closer points matter more
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5, weights="distance"))
    ]),

    # Random Forest uses lots of decision trees. It doesn't need scaled data
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    ),
}

# Creating the voting ensemble
# "Soft" voting means they look at the probabilities, not just the final label
voting_model = VotingClassifier(
    estimators=[
        ("lr", models["Logistic Regression"]),
        ("knn", models["KNN"]),
        ("rf", models["Random Forest"])
    ],
    voting="soft"
)

# Including the voting ensemble in evaluation
models["Voting Classifier"] = voting_model

# Empty list to store model performance scores
results = []

# Loop through models to train and evaluate each one
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    # Save scores to results list
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "F1-score": f1
    })

    # Print out the quick scores to the console
    print(f"{model_name} Accuracy:", accuracy)
    print(f"{model_name} F1-score:", f1)

    # Print a full breakdown showing precision and recall for both individual classes
    print(classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_
    ))


results_df = pd.DataFrame(results)

# Sort the table so the model with the highest F1-score is right at the top
results_df = results_df.sort_values(by="F1-score", ascending=False)

# Show the complete scoreboard in the console
print("\nFinal model comparison (binary):")
print(results_df)

# Save the final scores
results_df.to_csv("model_results_binary.csv", index=False)
print("Results saved as model_results_binary.csv")

# Confusion matrix for best model (Logistic Regression)
print("\nCreating confusion matrix for Logistic Regression...")

lr_model = models["Logistic Regression"]
lr_predictions = lr_model.predict(X_test)

# Calculate the grid of True/False Positives and Negatives
cm = confusion_matrix(y_test, lr_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

# Plot the matrix using a green color map
disp.plot(cmap="Greens")
plt.title("Confusion Matrix - Binary Logistic Regression")
plt.tight_layout()
plt.savefig("confusion_matrix_binary.png", dpi=300)
plt.show()

print("Saved: confusion_matrix_binary.png")

# Comparison chart: original 4-class vs new binary
print("\nCreating comparison chart...")

original_f1 = 0.5781  # from our original model_training_feature_engineering.py

# Setting up the text labels that will go along the bottom of our bar chart
labels = ["Original\n4-class LR", "Binary\nLogistic Regression", "Binary\nRandom Forest", "Binary\nKNN",
          "Voting\nClassifier"]

# Matching scores out of our sorted results dataframe to match the labels
f1_scores = [
    original_f1,
    results_df[results_df["Model"] == "Logistic Regression"]["F1-score"].values[0],
    results_df[results_df["Model"] == "Random Forest"]["F1-score"].values[0],
    results_df[results_df["Model"] == "KNN"]["F1-score"].values[0],
    results_df[results_df["Model"] == "Voting Classifier"]["F1-score"].values[0],
]

# Setting custom colors
colors = ["#5b8dd9", "#2ecc71", "#2ecc71", "#2ecc71", "#9b59b6"]

plt.figure(figsize=(10, 5))
bars = plt.bar(labels, f1_scores, color=colors, width=0.5)

# Draw a red dotted line across the chart at our old 58% baseline so we can spot improvements instantly
plt.axhline(y=original_f1, color="red", linestyle="--", linewidth=1.2, label=f"Original baseline ({original_f1})")

plt.ylim(0, 1.0)
plt.ylabel("F1-score")
plt.title("Before vs After: 4-Class vs Binary Classification")

# Add the value labels right above the top edge of each bar
for bar, val in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
             f"{val:.4f}", ha="center", fontsize=10)

# Add the legend box so people know what the red dotted line stands for
plt.legend()
plt.tight_layout()
plt.savefig("improvement_comparison.png", dpi=300)
plt.show()

print("Saved: improvement_comparison.png")

# Finding the highest scoring model
best_model_name = max(results, key=lambda item: item["F1-score"])["Model"]
print(f"\nWinner selected for export: {best_model_name}")

# Extracting and saving the winning model pipeline
best_model_pipeline = models[best_model_name]

# Save pipeline components and the needed feature metadata
joblib.dump(best_model_pipeline, "burnout_pipeline.pkl")
joblib.dump(label_encoder, "burnout_label_encoder.pkl")
# Exporting the list of training feature names
# We save this so another system knows the exact columns and order the model needs to work properly
joblib.dump(X_train.columns.tolist(), "model_features.pkl")

print("saved: burnout_label_encoder.pkl")
print("Saved: burnout_pipeline.pkl")
print("Saved: model_features.pkl")

print("\nDone!")
