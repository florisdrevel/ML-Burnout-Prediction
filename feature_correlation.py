import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load the cleaned and engineered dataset
df = pd.read_csv("company_data.csv")
# Select only numeric columns — can't correlate text/categorical
numeric_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix — every feature vs every other feature
corr_matrix = numeric_df.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    annot=True,  # Show the correlation number in each cell
    fmt='.2f',  # Round to 2 decimal places
    cmap='coolwarm',  # Red = positive, blue = negative
    linewidths=0.5,
    vmin=-1, vmax=1,
    annot_kws={'size': 6}
)
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
