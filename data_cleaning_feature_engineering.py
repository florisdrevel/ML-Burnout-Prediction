import pandas as pd

# Loading the original dataset
print("Loading original dataset...")

df = pd.read_csv("mental_health_burnout_tech_2026.csv", sep=";")

# Showing some general information to begin the analysis
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

# Removing the parent features
df = df.drop(columns=["salary_usd", "country_avg_salary"])


# Remove unwanted columns due to lack of signal or data leakage
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


# Now for the feauture engineering

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
# print results
print("Cleaned dataset shape:", df.shape)
print("Remaining columns:")
print(df.columns)
# save to the final dataset
df.to_csv("company_data.csv", index=False)

print("Engineered dataset saved as company_data.csv")
