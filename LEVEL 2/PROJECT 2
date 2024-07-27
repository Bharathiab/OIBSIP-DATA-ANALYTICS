import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv("C:\\Users\\S.Bharathi\\Downloads\\WineQT.csv")

# Drop unnecessary column
df = df.drop(columns=['Id'])

# Separate features and target variable
X = df.drop(columns=['quality'])
y = df['quality']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Initialize and train the SGD model
sgd_model = SGDClassifier(random_state=42)
sgd_model.fit(X_train, y_train)

# Initialize and train the SVC model
svc_model = SVC(random_state=42)
svc_model.fit(X_train, y_train)

# Make predictions and evaluate
rf_predictions = rf_model.predict(X_test)
sgd_predictions = sgd_model.predict(X_test)
svc_predictions = svc_model.predict(X_test)

# Print classification reports with zero_division parameter
print("Random Forest Classifier Report:")
print(classification_report(y_test, rf_predictions, zero_division=0))
print("Accuracy:", accuracy_score(y_test, rf_predictions))

print("\nSGD Classifier Report:")
print(classification_report(y_test, sgd_predictions, zero_division=0))
print("Accuracy:", accuracy_score(y_test, sgd_predictions))

print("\nSVC Classifier Report:")
print(classification_report(y_test, svc_predictions, zero_division=0))
print("Accuracy:", accuracy_score(y_test, svc_predictions))

# Visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot of features
sns.pairplot(df[['fixed acidity', 'volatile acidity', 'citric acid', 'density', 'quality']], hue='quality')
plt.show()

# Distribution of alcohol content
sns.histplot(df['alcohol'], kde=True)
plt.title('Distribution of Alcohol Content')
plt.show()
