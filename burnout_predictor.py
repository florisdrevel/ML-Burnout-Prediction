import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# This script lets a company use our trained model on their own employee data
# They need to fill in company_data.csv with their employees
# The script will then predict who is at risk of burnout


# Load our training dataset
print("Loading training data...")

df = pd.read_csv("cleaned_burnout_dataset_engineered.csv")

print("Training dataset shape:", df.shape)


# Separate features and target
X = df.drop(columns=["burnout_level"])
y = df["burnout_level"]

# Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Save the column names so we can match them later with the company data
training_columns = X.columns

# Use binary target: At Risk or Not At Risk
y_binary = y.map(lambda x: "At Risk" if x in ["High", "Severe"] else "Not At Risk")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_binary)

print("Target classes:", label_encoder.classes_)


# Train the model on our dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X_train, y_train)

print("Model trained successfully.")


# Load the company's employee data
print("\nLoading company data...")

company_df = pd.read_csv("company_data.csv")

# Clean up any invisible characters in column names (can happen when saving from Excel)
company_df.columns = company_df.columns.str.strip()

print("Number of employees:", len(company_df))


# Save employee IDs if they exist, so we can show them in the results
if "employee_id" in company_df.columns:
    employee_ids = company_df["employee_id"]
    company_df = company_df.drop(columns=["employee_id"])
else:
    employee_ids = ["Employee " + str(i + 1) for i in range(len(company_df))]


# Apply the same salary normalisation we did in data cleaning
company_df["country_avg_salary"] = company_df.groupby("country")["salary_usd"].transform("mean")
company_df["salary_relative"] = company_df["salary_usd"] / company_df["country_avg_salary"]
company_df = company_df.drop(columns=["salary_usd", "country_avg_salary"])


# Apply the same feature engineering we did before
company_df["work_sleep_ratio"] = company_df["work_hours_per_week"] / company_df["sleep_hours_per_night"]
company_df["meeting_workload_ratio"] = company_df["meetings_per_day"] / company_df["work_hours_per_week"]
company_df["work_life_risk"] = company_df["work_hours_per_week"] / company_df["work_life_balance_score"]
company_df["support_average"] = (
    company_df["manager_support_score"] +
    company_df["social_support_score"] +
    company_df["autonomy_score"]
) / 3
company_df["pressure_support_ratio"] = company_df["deadline_pressure_score"] / company_df["support_average"]


# Encode categorical columns the same way as training data
company_df = pd.get_dummies(company_df, drop_first=True)

# Make sure the company data has the same columns as the training data
# Some columns might be missing if the company has fewer categories
company_df = company_df.reindex(columns=training_columns, fill_value=0)


# Run predictions
print("\nRunning predictions...")

predictions = model.predict(company_df)
prediction_labels = label_encoder.inverse_transform(predictions)

# Get the probability for each class
# predict_proba returns a probability for each class (At Risk and Not At Risk)
# We take the probability of whichever class was predicted as the confidence score
probabilities = model.predict_proba(company_df)
confidence_scores = probabilities.max(axis=1)

# Round to a percentage
confidence_pct = (confidence_scores * 100).round(1)


# Save results
results = pd.DataFrame({
    "employee_id": employee_ids,
    "burnout_risk": prediction_labels,
    "confidence": confidence_pct
})

# Sort so At Risk employees show up first, then by confidence (highest first)
# This way HR sees the most urgent cases at the top
results = results.sort_values(
    by=["burnout_risk", "confidence"],
    ascending=[True, False]
)

results.to_csv("burnout_predictions.csv", index=False)

print("\nResults saved to burnout_predictions.csv")

# Save a separate file with only the At Risk employees
at_risk = results[results["burnout_risk"] == "At Risk"]

at_risk.to_csv("high_risk_employees.csv", index=False)

print("High risk employees saved to high_risk_employees.csv")
print("\nSummary:")
print(results["burnout_risk"].value_counts())
