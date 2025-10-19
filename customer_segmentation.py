import pandas as pd

# Load dataset
data = pd.read_csv('data/Mall_Customers.csv')

# Display first 5 rows
print("First 5 rows of dataset:")
print(data.head())

# Show summary info
print("\nData Info:")
print(data.info())

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram for Age, Annual Income, and Spending Score
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.histplot(data['Age'], bins=15, kde=True, color='green')
plt.title('Age Distribution')

plt.subplot(1, 3, 2)
sns.histplot(data['Annual Income (k$)'], bins=15, kde=True, color='blue')
plt.title('Annual Income Distribution')

plt.subplot(1, 3, 3)
sns.histplot(data['Spending Score (1-100)'], bins=15, kde=True, color='orange')
plt.title('Spending Score Distribution')

plt.tight_layout()
plt.savefig('outputs/Spending_Score_Distribution.png')
plt.show()

from sklearn.cluster import KMeans

# Select relevant features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# View first few rows
print("\nFeatures for clustering:")
print(X.head())

wcss = []  # Within Cluster Sum of Squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method - Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('outputs/Elbow_Method.png')
plt.show()

# Assuming optimal clusters = 5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

print("\nCluster Labels Assigned:")
print(data.head())

import seaborn as sns

plt.figure(figsize=(8,6))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set1',
    data=data,
    s=100
)
plt.title('Customer Segments (K-Means Clustering)')
plt.savefig('outputs/customer_segments.png')
plt.show()

import os

# Define the directory path
output_dir = 'outputs'

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Now, save the file
data.to_csv(os.path.join(output_dir, 'clustered_customers.csv'), index=False)

# Save the clustered data into outputs folder
data.to_csv('outputs/clustered_customers.csv', index=False)
print("\nClustered results saved to outputs/clustered_customers.csv")

print("\nCluster Profiles:")
cluster_profile = data.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(cluster_profile)


