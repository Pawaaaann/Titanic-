# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load the Titanic dataset
df = pd.read_csv('train.csv')  # Ensure 'train.csv' is in the same directory as this script

# 2. Explore the dataset
print("First 5 rows of the dataset:")
print(df.head())  # Display first 5 rows

# Basic info
print("\nBasic Information about the dataset:")
print(df.info())  # Info about columns, data types, and missing values

# Summary statistics for numerical columns
print("\nSummary statistics for numerical columns:")
print(df.describe())

# Checking for missing values in each column
print("\nMissing values in each column:")
print(df.isnull().sum())

# 3. Data Cleaning
# Handle missing values
# Fill missing Age with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)
# Fill missing Embarked with mode (most frequent value)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop rows with missing 'Survived' or 'Pclass'
df.dropna(subset=['Survived', 'Pclass'], inplace=True)

# 4. Feature Engineering
# Convert categorical columns to numerical values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Male = 0, Female = 1
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # C = 0, Q = 1, S = 2

# 5. Exploratory Data Analysis (EDA)
# Visualizing the distribution of survival
sns.countplot(x='Survived', data=df)
plt.title('Survival Distribution')
plt.show()

# Survival by Sex
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Sex')
plt.show()

# Survival by Pclass
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival by Pclass')
plt.show()

# Age vs Survival
sns.histplot(df[df['Survived'] == 1]['Age'], kde=True, color='green', label='Survived')
sns.histplot(df[df['Survived'] == 0]['Age'], kde=True, color='red', label='Did not Survive')
plt.legend()
plt.title('Age vs Survival')
plt.show()

# 6. Prepare the data for training
# Selecting features and target variable
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]  # Features
y = df['Survived']  # Target

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Building the Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Logistic Regression model
model.fit(X_train, y_train)  # Train the model

# 8. Make predictions on the test set
y_pred = model.predict(X_test)

# 9. Evaluate the model

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Did not Survive', 'Survived'], yticklabels=['Did not Survive', 'Survived'])
plt.title('Confusion Matrix')
plt.show()
