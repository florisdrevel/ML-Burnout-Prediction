import pandas as pd

print("Loading original dataset...")

df = pd.read_csv("/Users/ailinbergetun/documents/Exam_Project/mental_health_burnout_tech_2026.csv")

print("Original dataset shape:", df.shape)
print(df.head())
print(df.info())
print("Missing values:")
print(df.isnull().sum())

print("Burnout level distribution:")
print(df["burnout_level"].value_counts())


# Normalise salary by country
print("Normalising salary by country...")

df["country_avg_salary"] = df.groupby("country")["salary_usd"].transform("mean")
df["salary_relative"] = df["salary_usd"] / df["country_avg_salary"]

df = df.drop(columns=["salary_usd", "country_avg_salary"])


# Remove unwanted columns
columns_to_remove = [
    "employee_id",
    "years_at_company",
    "phq9_category",
    "gad7_category",
    "burnout_score",
    "stress_score",
    "phq9_score",
    "gad7_score",
    "seeks_mental_health_support",
    "job_change_intention"
]

df = df.drop(columns=columns_to_remove)

print("Cleaned dataset shape:", df.shape)
print("Remaining columns:")
print(df.columns)

df.to_csv("cleaned_burnout_dataset.csv", index=False)

print("Cleaned dataset saved as cleaned_burnout_dataset.csv")