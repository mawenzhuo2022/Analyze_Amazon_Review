# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/15 23:30
# @Function: K-means clustering with the optimal number of clusters determined via the elbow method.

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 设置为非GUI后端
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_data():
    """Load training and testing data from CSV files.
    加载训练和测试数据集。
    """
    X_train = pd.read_csv('app/backend/dat/analyze/cleaned_data/X_train.csv')  # Load training data from CSV files
    Y_train = pd.read_csv('app/backend/dat/analyze/cleaned_data/Y_train.csv')  # Load labels of training data from CSV files
    X_test = pd.read_csv('app/backend/dat/analyze/cleaned_data/X_test.csv')    # Load testing data from CSV files
    Y_test = pd.read_csv('app/backend/dat/analyze/cleaned_data/Y_test.csv')    # Load labels of testing data from CSV files
    return X_train, Y_train, X_test, Y_test

def plot_elbow_method(data):
    """Generate elbow plot to determine optimal number of clusters for k-means.
    生成肘部法则图，用以确定k-means的最佳聚类数量。
    """
    inertias = []                        # List to store inertia values for each number of clusters
    k_range = range(1, 11)               # Testing different number of clusters from 1 to 10
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)  # Create KMeans model with given number of clusters
        kmeans.fit(data)                 # Fit the data
        inertias.append(kmeans.inertia_)  # Append the inertia value to the list

    plt.figure(figsize=(8, 4))           # Create the plot window
    plt.plot(k_range, inertias, marker='o')  # Plot the elbow method
    plt.title('Elbow Method For Optimal k')  # Set the title
    plt.xlabel('Number of clusters')     # Set the X-axis label
    plt.ylabel('Inertia')                # Set the Y-axis label
    plt.xticks(k_range)                  # Set the X-axis ticks
    plt.grid(True)                       # Show grid lines
    plt.savefig('elbow_method.png')      # Save the plot to a file
    plt.close()                          # Close the plot window to free up resources

def fit_model(data, k=3):
    """Fit the KMeans model using a pre-determined optimal k value.
    使用预先确定的最优k值拟合KMeans模型。
    """
    kmeans = KMeans(n_clusters=k, random_state=42)  # Create KMeans model with pre-determined optimal number of clusters
    clusters = kmeans.fit_predict(data)            # Cluster the data
    return clusters, kmeans

def save_cluster_data(data, clusters, filepath):
    """Save the clustered data to a CSV file.
    将聚类数据保存到 CSV 文件中。
    """
    # Add cluster labels to the dataframe
    data['Cluster'] = clusters
    data.to_csv(filepath, index=False)  # Save the data to a CSV file without saving the index
    print(f"Cluster data saved to {filepath}")  # Print the path where the file is saved

def main():
    """Main function to run the clustering analysis.
    主函数运行聚类分析。
    """
    X_train, Y_train, X_test, Y_test = load_data()  # Load the data
    plot_elbow_method(X_train)                       # Plot the elbow method to determine the optimal number of clusters
    clusters, kmeans = fit_model(X_train)            # Cluster the training data
    save_cluster_data(X_train, clusters, "app/backend/dat/analyze/cluster/cluster.csv")  # Save the clustered result

if __name__ == "__main__":
    main()
