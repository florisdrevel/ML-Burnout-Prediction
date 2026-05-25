import pandas as pd

print("Loading model results...")

results = pd.read_csv("model_results.csv")

print("\nFinal model comparison:")
print(results)

best_model = results.iloc[0]

print("\nBest model:")
print(best_model["Model"])

print("\nBest accuracy:")
print(best_model["Accuracy"])

print("\nBest F1-score:")
print(best_model["F1-score"])