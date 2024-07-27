# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv('C:\\Users\\S.Bharathi\\Downloads\\ifood_df.csv')

# Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Data Cleaning
# Handle missing values (example: fill with median or drop)
df.fillna(df.median(), inplace=True)

# Feature Engineering (example: using Recency, MntTotal, and Age for clustering)
features = df[['Recency', 'MntTotal', 'Age']].dropna()

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-means clustering with the chosen number of clusters (e.g., 4)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=df['Recency'], y=df['MntTotal'], hue=df['Cluster'], palette='viridis')
plt.title('Customer Segments')
plt.xlabel('Recency')
plt.ylabel('Total Expenditure')
plt.legend(title='Cluster')
plt.show()

# Calculate silhouette score for clustering evaluation
silhouette_avg = silhouette_score(scaled_features, df['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')

# Analyze segments
segment_analysis = df.groupby('Cluster').mean()
print(segment_analysis)
