import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Load data

df = pd.read_csv("company_data.csv")

print("Loaded engineered dataset:")
print(df.shape)


# Figure 1, burnout level distribution

plt.figure(figsize=(8, 5))
df["burnout_level"].value_counts().plot(kind="bar")

plt.title("Burnout Level Distribution")
plt.xlabel("Burnout Level")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("burnout_distribution.png", dpi=300)
plt.show()


# Figure 2, correlation heatmap

numeric_df = df.select_dtypes(include=["number"])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(14, 12))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    vmin=-1,
    vmax=1,
    annot_kws={"size": 6}
)

plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()

plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()


# Figure 3, model comparison

results = pd.read_csv("model_results.csv")

plt.figure(figsize=(9, 5))
plt.bar(results["Model"], results["F1-score"])

plt.title("Model Comparison by F1-score")
plt.xlabel("Model")
plt.ylabel("F1-score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig("model_comparison_f1.png", dpi=300)
plt.show()


# Prepping data for confusion matrix and feature importance

X = df.drop(columns=["burnout_level"])
y = df["burnout_level"]

X = pd.get_dummies(X, drop_first=True)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Figure 4, confusion matrix of the logistic regression model

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

cm = confusion_matrix(y_test, logistic_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()

plt.savefig("confusion_matrix_logistic_regression.png", dpi=300)
plt.show()


# Figure 5, feature importance of the random forest model

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

top_features = feature_importance.head(10)

plt.figure(figsize=(9, 5))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.gca().invert_yaxis()

plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()

plt.savefig("feature_importance.png", dpi=300)
plt.show()


print("All visualisations created from engineered dataset.")
