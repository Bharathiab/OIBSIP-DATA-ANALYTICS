import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time

# Load the data
data = pd.read_csv('C:/Users/S.Bharathi/Downloads/creditcard.csv (1)/creditcard.csv')

# Feature Engineering
data['log_Amount'] = np.log1p(data['Amount'])
X = data.drop(['Class'], axis=1)
y = data['Class']

# Data Distribution Visualization
plt.figure(figsize=(20, 30))
num_cols = len(X.columns)
for i, column in enumerate(X.columns, 1):
    plt.subplot((num_cols // 5) + 1, 5, i)
    sns.histplot(data[column], bins=30, kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()

# Correlation Matrix Visualization
plt.figure(figsize=(15, 10))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Anomaly Detection
iso_forest = IsolationForest(contamination=0.01)
anomalies = iso_forest.fit_predict(X)
data['Anomaly'] = anomalies

# Anomaly Visualization
plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['Amount'], c=data['Anomaly'], cmap='coolwarm', alpha=0.6)
plt.title('Anomaly Detection')
plt.xlabel('Index')
plt.ylabel('Amount')
plt.show()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Machine Learning Models
# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
print("Logistic Regression:\n", classification_report(y_test, lr_pred))

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)
print("Decision Tree:\n", classification_report(y_test, dt_pred))

# Neural Network
nn_model = MLPClassifier(hidden_layer_sizes=(30,30,30), max_iter=500)
nn_model.fit(X_train_scaled, y_train)
nn_pred = nn_model.predict(X_test_scaled)
print("Neural Network:\n", classification_report(y_test, nn_pred))

# Confusion Matrix Visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
models = ['Logistic Regression', 'Decision Tree', 'Neural Network']
preds = [lr_pred, dt_pred, nn_pred]
for ax, model, pred in zip(axes, models, preds):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{model} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
plt.tight_layout()
plt.show()

# Simulated Real-Time Monitoring
def monitor_transactions(new_data):
    # Preprocess new data
    new_data_scaled = scaler.transform(new_data)
    
    # Predict using the trained model
    predictions = nn_model.predict(new_data_scaled)
    
    # Flag transactions if necessary
    flagged_transactions = new_data[predictions == 1]
    
    return flagged_transactions

# Simulate real-time data streaming with smaller batches
for _ in range(5):  # Run for 5 iterations
    new_data = pd.DataFrame(np.random.randn(10, X.shape[1]), columns=X.columns)  # Simulate 10 new transactions
    flagged = monitor_transactions(new_data)
    print("Flagged Transactions:\n", flagged)
    time.sleep(1)  # Pause for 1 second before checking again

