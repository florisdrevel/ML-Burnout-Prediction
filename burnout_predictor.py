import joblib
import pandas as pd

# Loading saved model components
model_pipeline = joblib.load("burnout_pipeline.pkl")
label_encoder = joblib.load("burnout_label_encoder.pkl")
model_features = joblib.load("model_features.pkl")

# Loading the new production dataset
new_df = pd.read_csv("company_data.csv")

# Dropping identifiers and labels to isolate features
columns_to_drop = ["burnout_level", "name", "employee_id"]
X_new = new_df.drop(columns=[col for col in columns_to_drop if col in new_df.columns])

# Converting text features to dummy columns
X_new = pd.get_dummies(X_new, drop_first=True)

# Aligning new columns with original training features
missing_cols = set(model_features) - set(X_new.columns)
for col in missing_cols:
    X_new[col] = 0

# Ordering the final features exactly as expected
X_new = X_new[model_features]

# Running numeric and text classification steps
numeric_predictions = model_pipeline.predict(X_new)
text_predictions = label_encoder.inverse_transform(numeric_predictions)

# Mapping text predictions back onto production data
new_df["predicted_burnout_level"] = text_predictions
new_df.to_csv("predictions_output.csv", index=False)

print("Predictions saved to predictions_output.csv")
