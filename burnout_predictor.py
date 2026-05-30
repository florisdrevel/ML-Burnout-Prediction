import joblib
import pandas as pd

# Loading saved model components
model_pipeline = joblib.load("burnout_pipeline.pkl")
label_encoder = joblib.load("burnout_label_encoder.pkl")
model_features = joblib.load("model_features.pkl")

# Load the Data
df = pd.read_csv("company_data.csv")

# Dropping ID's
columns_to_drop = ["name", "employee_id"]
X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Converting text features to dummy columns
X = pd.get_dummies(X, drop_first=True)

# Aligning columns with original training features
missing_cols = set(model_features) - set(X.columns)
for col in missing_cols:
    X[col] = 0

# Ordering the final features exactly as expected
X = X[model_features]

numeric_predictions = model_pipeline.predict(X)
y_pred = label_encoder.inverse_transform(numeric_predictions)

# Because Pandas keeps the original row order unchanged throughout the process,
# we can re-attach the predictions array back as a new column
df["predicted_burnout_level"] = y_pred
df.to_csv("predictions_output.csv", index=False)

print("Predictions saved to predictions_output.csv")
