import matplotlib.pyplot as plt
import pandas as pd

# use the scores of the models to create a dataframe
data = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM", "XGBoost", "Random Forest", "KNN"],
    "Before": [0.5774, 0.5689, 0.5665, 0.5203, 0.3816],
    "After": [0.5781, 0.5700, 0.5662, 0.5396, 0.4038]
})

x = range(len(data))

# plot before and after on 1 chart
plt.figure(figsize=(9, 5))
plt.plot(x, data["Before"], marker='o', label="Before")
plt.plot(x, data["After"], marker='o', label="After")

plt.xticks(x, data["Model"], rotation=30)
plt.ylabel("F1-score")
plt.title("Model Performance Before and After Feature Engineering")
plt.legend()

plt.tight_layout()
plt.savefig("feature_engineering_comparison.png")
plt.show()
