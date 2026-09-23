import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = "wine_dataset.csv"
OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS - WINE DATASET")
print("=" * 60)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATASET INFORMATION ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nData Types:")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
print(missing)
print("Total missing values:", missing.sum())

print("\n--- DUPLICATES ---")
print("Duplicate rows:", df.duplicated().sum())

print("\n--- STATISTICAL SUMMARY ---")
summary = df.describe()
print(summary)
summary.to_csv(os.path.join(OUTPUT_DIR, "statistical_summary.csv"))

print("\n--- WINE CLASS DISTRIBUTION ---")
print(df["target_name"].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="target_name")
plt.title("Wine Class Distribution")
plt.xlabel("Wine Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=300)
plt.show()

numeric_features = df.select_dtypes(include=np.number).columns.tolist()
numeric_features.remove("target")

df[numeric_features].hist(figsize=(16, 14), bins=20)
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_distributions.png"), dpi=300)
plt.show()

plt.figure(figsize=(16, 8))
sns.boxplot(data=df[numeric_features])
plt.xticks(rotation=70, ha="right")
plt.title("Boxplot of Numerical Features")
plt.xlabel("Features")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "boxplots.png"), dpi=300)
plt.show()

corr = df[numeric_features + ["target"]].corr()
target_corr = corr["target"].drop("target").sort_values(ascending=False)

print("\n--- CORRELATION WITH TARGET ---")
print(target_corr)
corr.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))

plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=300)
plt.show()

print("\n--- TOP CORRELATED FEATURES ---")
top = target_corr.abs().sort_values(ascending=False).head(5)
for feature in top.index:
    print(feature, ":", round(target_corr[feature], 3))

plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x="flavanoids", y="od280/od315_of_diluted_wines", hue="target_name", style="target_name", s=80)
plt.title("Flavanoids vs OD280/OD315 by Wine Class")
plt.xlabel("Flavanoids")
plt.ylabel("OD280/OD315 of Diluted Wines")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_scatter_plot.png"), dpi=300)
plt.show()

top_corr = target_corr.abs().sort_values(ascending=False).head(8)
plt.figure(figsize=(10, 6))
top_corr.sort_values().plot(kind="barh")
plt.title("Top Features by Absolute Correlation with Target")
plt.xlabel("Absolute Correlation")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_correlated_features.png"), dpi=300)
plt.show()

group_means = df.groupby("target_name")[numeric_features].mean()
print("\n--- AVERAGE FEATURE VALUES BY WINE CLASS ---")
print(group_means.round(2))
group_means.to_csv(os.path.join(OUTPUT_DIR, "group_wise_means.csv"))

highest_positive = target_corr.idxmax()
highest_negative = target_corr.idxmin()

print("\n" + "=" * 60)
print("KEY EDA INSIGHTS")
print("=" * 60)
print(f"1. Strongest positive correlation with encoded target: {highest_positive} ({target_corr[highest_positive]:.3f})")
print(f"2. Strongest negative correlation with encoded target: {highest_negative} ({target_corr[highest_negative]:.3f})")
print(f"3. Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"4. Missing values: {missing.sum()}.")
print(f"5. Duplicate rows: {df.duplicated().sum()}.")
print("6. Histograms show numerical feature distributions.")
print("7. Boxplots show spread and possible outliers.")
print("8. The heatmap shows relationships between numerical variables.")
print("9. The scatter plot compares selected features across wine classes.")

df.to_csv(os.path.join(OUTPUT_DIR, "analyzed_wine_dataset.csv"), index=False)
print("\nEDA analysis completed successfully!")
print("All results are saved in the 'eda_outputs' folder.")
