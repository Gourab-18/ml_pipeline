"""
Generate key EDA plots for the notebook.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))
from data.loader import DataLoader

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

# Load data
loader = DataLoader("configs/schema.yaml")
df = loader.load_data("data/sample.csv")

# Create output directory
import os
os.makedirs("notebooks/plots", exist_ok=True)

# 1. Missing values heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('notebooks/plots/missing_values_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Target distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram
axes[0].hist(df['churn_probability'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0].set_title('Target Variable Distribution')
axes[0].set_xlabel('Churn Probability')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['churn_probability'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["churn_probability"].mean():.3f}')
axes[0].legend()

# Bar plot of class counts
class_counts = df['churn_probability'].value_counts().sort_index()
axes[1].bar(class_counts.index, class_counts.values, color=['lightcoral', 'lightblue'])
axes[1].set_title('Class Distribution')
axes[1].set_xlabel('Churn (0=No, 1=Yes)')
axes[1].set_ylabel('Count')
axes[1].set_xticks([0, 1])

# Add count labels on bars
for i, v in enumerate(class_counts.values):
    axes[1].text(i, v + 5, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('notebooks/plots/target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Feature correlations
numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'churn_probability']
if len(numeric_cols) > 0:
    correlations = df[numeric_cols + ['churn_probability']].corr()['churn_probability'].drop('churn_probability').sort_values(ascending=False, key=abs)
    
    plt.figure(figsize=(12, 8))
    correlations.plot(kind='barh', color='skyblue')
    plt.title('Feature Correlations with Target Variable')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.axvline(0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('notebooks/plots/feature_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()

# 4. Categorical features
categorical_features = ['gender', 'location_country', 'subscription_type']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, feature in enumerate(categorical_features):
    if feature in df.columns:
        value_counts = df[feature].value_counts()
        
        axes[i].bar(range(len(value_counts)), value_counts.values, color='lightcoral')
        axes[i].set_title(f'{feature} Distribution')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Count')
        axes[i].set_xticks(range(len(value_counts)))
        axes[i].set_xticklabels(value_counts.index, rotation=45)
        
        # Add count labels on bars
        for j, v in enumerate(value_counts.values):
            axes[i].text(j, v + 1, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('notebooks/plots/categorical_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

print("Plots generated successfully!")
print("Generated files:")
print("- notebooks/plots/missing_values_heatmap.png")
print("- notebooks/plots/target_distribution.png")
print("- notebooks/plots/feature_correlations.png")
print("- notebooks/plots/categorical_distributions.png")
