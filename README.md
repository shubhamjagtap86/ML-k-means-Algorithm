🌸 K-Means Clustering on Iris Dataset

This project demonstrates how to perform unsupervised learning using the K-Means clustering algorithm to group similar observations from the famous Iris dataset.
The implementation includes data loading, visualization, clustering, and evaluation — all performed in the notebook Means Clustering (1).ipynb.

📁 Dataset

The dataset used in this project is the Iris Dataset, one of the most popular datasets in pattern recognition and clustering.

Dataset Details:

Total records: 150

Features: 4 numerical features and 1 categorical target (species)

Target variable: Species (for reference only; not used in clustering)

Feature	Description
sepal_length	Length of the sepal (cm)
sepal_width	Width of the sepal (cm)
petal_length	Length of the petal (cm)
petal_width	Width of the petal (cm)
species	Iris type (Setosa, Versicolor, Virginica)

📦 Dataset Source:
UCI Machine Learning Repository – Iris Dataset

🧰 Libraries Used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

🧹 Data Preprocessing

Loaded dataset into a Pandas DataFrame.

Selected numerical features for clustering: sepal_length, sepal_width, petal_length, petal_width.

Handled missing or invalid values (if any).

Visualized feature distributions using pairplots and heatmaps.

🧮 K-Means Model Building

Initialized KMeans model with varying numbers of clusters (k = 1 to 10).

Computed WCSS (Within-Cluster Sum of Squares) for each k-value.

Determined the optimal number of clusters using the Elbow Method.

Trained final KMeans model with the best cluster count (commonly k = 3).

Predicted cluster labels and visualized results.

📈 Model Evaluation

Metrics Used:

WCSS (Elbow Method)

Silhouette Score – measures how well each point fits its assigned cluster.

Example:

score = silhouette_score(X, labels)
print("Silhouette Score:", score)

📊 Visualization

Scatter plots showing cluster separations.

Pairplots for feature comparisons.

Cluster centroids highlighted for interpretation.

Example:

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='X', s=200)
plt.title("K-Means Clustering - Iris Data")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

🚀 How to Run

Open the notebook:

jupyter notebook "Means Clustering (1).ipynb"


Run each cell in sequence:

Load and explore the dataset

Apply the K-Means algorithm

Visualize the clusters

Evaluate model performance

🧩 Insights

Optimal cluster count determined as 3, corresponding closely to the three Iris species.

Clusters formed are well-separated based on petal length and petal width, indicating strong discriminative power.

The Silhouette Score confirms a good clustering structure.