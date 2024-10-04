# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset (Titanic dataset)
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic_df = pd.read_csv(url)

# 2. Data Exploration
print(titanic_df.head())        # Show first 5 rows
print(titanic_df.info())        # Show data types and null values
print(titanic_df.describe())    # Summary statistics

# 3. Data Cleaning
# Handling missing values (fill missing age with median, drop missing embarked)
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df.dropna(subset=['Embarked'], inplace=True)

# Convert 'Sex' and 'Embarked' to categorical variables
titanic_df['Sex'] = titanic_df['Sex'].astype('category').cat.codes
titanic_df['Embarked'] = titanic_df['Embarked'].astype('category').cat.codes

# 4. Data Analysis
# Find survival rate by gender
survival_rate_by_gender = titanic_df.groupby('Sex')['Survived'].mean()

# Print the survival rates
print(f"Survival Rate by Gender:\n{survival_rate_by_gender}")

# 5. Data Visualization
# Bar plot of survival rates by gender
plt.figure(figsize=(8, 6))
sns.barplot(x=['Female', 'Male'], y=survival_rate_by_gender)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.xlabel('Gender')
plt.show()

# 6. Further Analysis - Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = titanic_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Titanic Dataset')
plt.show()
