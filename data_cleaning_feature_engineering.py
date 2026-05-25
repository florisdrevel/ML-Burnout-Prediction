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


# FEATURE ENGINEERING

print("Creating new engineered features...")

# Workload compared to sleep
df["work_sleep_ratio"] = df["work_hours_per_week"] / df["sleep_hours_per_night"]

# Meetings compared to work hours
df["meeting_workload_ratio"] = df["meetings_per_day"] / df["work_hours_per_week"]

# Work-life risk indicator
df["work_life_risk"] = df["work_hours_per_week"] / df["work_life_balance_score"]

# Support average
df["support_average"] = (
    df["manager_support_score"] +
    df["social_support_score"] +
    df["autonomy_score"]
) / 3

# Pressure compared to support
df["pressure_support_ratio"] = df["deadline_pressure_score"] / df["support_average"]

print("New features created:")
print([
    "work_sleep_ratio",
    "meeting_workload_ratio",
    "work_life_risk",
    "support_average",
    "pressure_support_ratio"
])

print("Cleaned dataset shape:", df.shape)
print("Remaining columns:")
print(df.columns)

df.to_csv("cleaned_burnout_dataset_engineered.csv", index=False)

print("Engineered dataset saved as cleaned_burnout_dataset_engineered.csv")