Burnout Risk Predictor - README

Developed by Ailin Bergetun & Florean Drevel

Applied Machine Learning Exam Project 2025/2026
---------------------------------------------------------------------------------------------
WHAT THIS TOOL DOES
This tool predicts which employees are at risk of burnout.
It instantly loads our pre-trained model pipeline, processes your
employee data file, and saves a unified output file showing who is at risk.

---------------------------------------------------------------------------------------------
WHAT YOU NEED

Python installed with these libraries:

pandas

scikit-learn

joblib

If you don't have them, run this in your terminal:
pip install pandas scikit-learn joblib

These files all in the same folder:

burnout_predictor.py

burnout_pipeline.pkl

burnout_label_encoder.pkl

model_features.pkl

company_data.csv (your employee data - see below)

after you rename it from:

company_data_template.csv

---------------------------------------------------------------------------------------------
HOW TO PREPARE YOUR EMPLOYEE DATA

Use company_data_template.csv as your starting point.
Fill in your employees and save the file with the exact same name, or change it to simply company_data.csv
Make sure it is in the same folder as burnout_predictor.py.

The file needs these columns:
employee_id              - your internal employee ID
name                     - employee full name
age                      - age in years
gender                   - Male / Female / Non-binary
country                  - country of employment
job_role                 - job title
seniority_level          - Junior / Mid / Senior
years_experience         - total years of work experience
company_size             - Startup (1-50) / Small (51-200) / Medium (201-1000) / Large (1000+)
industry                 - industry sector
work_mode                - Remote / Hybrid / On-site
work_hours_per_week      - average hours worked per week
meetings_per_day         - average meetings per day
team_size                - number of people in their team
sleep_hours_per_night    - average sleep hours per night
exercise_days_per_week   - days per week they exercise
vacation_days_taken      - vacation days taken last year
therapy_access           - access to therapy? (1=yes, 0=no)
uses_therapy             - currently uses therapy? (1=yes, 0=no)
ai_tools_daily           - uses AI tools daily? (1=yes, 0=no)
manager_support_score    - score 10-100
work_life_balance_score  - score 10-100
job_satisfaction_score   - score 10-100
social_support_score     - score 10-100
deadline_pressure_score  - score 20-100
autonomy_score           - score 15-100
salary_relative          - dynamic base factor scale, dividing yearly income by country average.

---------------------------------------------------------------------------------------------
HOW TO RUN IT
Open your terminal or PyCharm

Navigate to the folder with the files

Run: python burnout_predictor.py

Wait for it to finish - it will print a confirmation message when done

---------------------------------------------------------------------------------------------
OUTPUT FILES
One unified file will be saved in the same folder:

predictions_output.csv

The complete original employee list including all names and IDs

Contains a new column: "predicted_burnout_level"

Each individual is clearly tagged as At Risk or Not At Risk

Use this to prioritize who to follow up with for check-ins

---------------------------------------------------------------------------------------------
IMPORTANT NOTE
This tool is a screening aid, not a diagnosis.
Use the results to start conversations and check in with employees.
Do not make decisions about anyone based solely on this output.
Employees should be aware the tool is being used.

---------------------------------------------------------------------------------------------
TECHNICAL NOTE
If your CSV file uses semicolons instead of commas as separators
(this can happen when saving from Excel on some systems), open
burnout_predictor.py and change the line reading the file from:

new_df = pd.read_csv("company_data.csv")

to:

new_df = pd.read_csv("company_data.csv", sep=";")
