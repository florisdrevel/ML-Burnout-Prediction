import pandas as pd

# Load the original dataset
df = pd.read_csv("company_data.csv")

# Drop all rows while keeping the column headers
df_template = df.iloc[0:0]

# Save the empty template to a new CSV file
df_template.to_csv("company_data_template.csv", index=False)

print("Template created successfully with 0 rows!")
