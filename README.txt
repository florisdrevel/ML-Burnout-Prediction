Burnout Risk Predictor - README
================================
Developed by Ailin Bergetun & Florean Drevel
Applied Machine Learning Exam Project 2025/2026


WHAT THIS TOOL DOES
--------------------
This tool predicts which employees are at risk of burnout.
You provide your employee data in a CSV file, run the script,
and it saves two output files showing who is at risk.


WHAT YOU NEED
--------------
1. Python installed with these libraries:
   - pandas
   - scikit-learn

   If you don't have them, run this in your terminal:
   pip install pandas scikit-learn

2. These files all in the same folder:
   - burnout_predictor.py
   - cleaned_burnout_dataset_engineered.csv
   - company_data.csv (your employee data - see below)


HOW TO PREPARE YOUR EMPLOYEE DATA
-----------------------------------
Use company_data_template.csv as your starting point.
Fill in your employees and save the file as company_data.csv.
Make sure it is in the same folder as burnout_predictor.py.

The file needs these columns:
   employee_id              - your internal employee ID
   age                      - age in years
   gender                   - Male / Female / Non-binary
   country                  - country of employment
   job_role                 - job title
   seniority_level          - Junior / Mid / Senior
   years_experience         - total years of work experience
   company_size             - Small / Medium / Large
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
   manager_support_score    - score 1-10
   work_life_balance_score  - score 1-10
   job_satisfaction_score   - score 1-10
   social_support_score     - score 1-10
   deadline_pressure_score  - score 1-10
   autonomy_score           - score 1-10
   salary_usd               - annual salary in USD


HOW TO RUN IT
--------------
1. Open your terminal or PyCharm
2. Navigate to the folder with the files
3. Run: python burnout_predictor.py
4. Wait for it to finish - it will print a summary when done


OUTPUT FILES
-------------
Two files will be saved in the same folder:

   burnout_predictions.csv
   - All employees with their predicted burnout risk
   - Sorted so At Risk employees appear first

   high_risk_employees.csv
   - Only the employees predicted as At Risk
   - Use this to prioritise who to follow up with


IMPORTANT NOTE
---------------
This tool is a screening aid, not a diagnosis.
Use the results to start conversations and check in with employees.
Do not make decisions about anyone based solely on this output.
Employees should be aware the tool is being used.


TECHNICAL NOTE
---------------
If your CSV file uses semicolons instead of commas as separators
(this can happen when saving from Excel on some systems), open
burnout_predictor.py and change line 63 from:

   company_df = pd.read_csv("company_data.csv")

to:

   company_df = pd.read_csv("company_data.csv", sep=";")
