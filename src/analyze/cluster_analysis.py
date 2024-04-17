# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/15 23:30
# @Function: K-means clustering with the optimal number of clusters determined via the elbow method.

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_data():
    """Load training and testing data from CSV files.
    加载训练和测试数据集。
    """
    X_train = pd.read_csv('../../dat/analyze/cleaned_data/X_train.csv')
    Y_train = pd.read_csv('../../dat/analyze/cleaned_data/Y_train.csv')
    X_test = pd.read_csv('../../dat/analyze/cleaned_data/X_test.csv')
    Y_test = pd.read_csv('../../dat/analyze/cleaned_data/Y_test.csv')
    return X_train, Y_train, X_test, Y_test

def plot_elbow_method(data):
    """Generate elbow plot to determine optimal number of clusters for k-means.
    生成肘部法则图，用以确定k-means的最佳聚类数量。
    """
    inertias = []
    k_range = range(1, 11)  # Testing different k values from 1 to 10. 测试从1到10的不同k值。
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(k_range, inertias, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

def fit_model(data, k=3):
    """Fit the KMeans model using a pre-determined optimal k value.
    使用预先确定的最优k值拟合KMeans模型。
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

def analyze_clusters(data, clusters):
    """Analyze and print cluster data.
    分析并打印聚类数据。
    """
    cluster_data = pd.concat([data, pd.DataFrame(clusters, columns=['Cluster'])], axis=1)
    for cluster in sorted(cluster_data['Cluster'].unique()):
        print(f"\nCluster {cluster} Top terms:")
        current_cluster_data = cluster_data[cluster_data['Cluster'] == cluster]
        word_freq = current_cluster_data.drop('Cluster', axis=1).mean().sort_values(ascending=False)
        print(word_freq.head())

def main():
    """Main function to run the clustering analysis.
    主函数运行聚类分析。
    """
    X_train, Y_train, X_test, Y_test = load_data()
    plot_elbow_method(X_train)
    clusters, kmeans = fit_model(X_train)
    analyze_clusters(X_train, clusters)

if __name__ == "__main__":
    main()
