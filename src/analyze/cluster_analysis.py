# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/15 23:30
# @Function:


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X_train = pd.read_csv('../../dat/analyze/cleaned_data/X_train.csv')
X_test = pd.read_csv('../../dat/analyze/cleaned_data/X_test.csv')
Y_train = pd.read_csv('../../dat/analyze/cleaned_data/Y_train.csv')
Y_test = pd.read_csv('../../dat/analyze/cleaned_data/Y_test.csv')

"""
观察肘部法则图： 运行 plot_elbow_method(X_train) 函数，这会生成一个肘部法则图，帮助你直观地看到每个 k 值对应的惯性（Inertia）变化情况。通常，随着 k 值的增加，惯性会减小，但是减小的速度会逐渐减缓。在图上找到拐点，即肘部，这通常对应着最佳的聚类数量。

评估聚类结果： 选择几个 k 值，比如 2、3、4、5，然后使用每个 k 值对应的聚类结果来评估模型的效果。你可以使用一些内部评估指标（如轮廓系数）或者外部指标（如与标签的匹配程度，如果有的话）来衡量每个 k 值的聚类效果。

调整参数并重复： 如果肘部法则图不明显，或者在评估中没有找到明显的最佳 k 值，可以尝试不同的参数设置或算法来寻找更好的聚类效果。这可能涉及到使用不同的初始化方法、迭代次数、距离度量等。

交叉验证： 如果数据量允许，可以考虑使用交叉验证来评估不同 k 值下的模型性能，这样可以更好地了解模型的泛化能力。
"""
def plot_elbow_method(data):
    inertias = []
    k_range = range(1, 11)  # 测试从1到10的不同k值
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    # 绘制肘部法则图
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, inertias, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.xticks(k_range)
    plt.grid(True)  # 添加网格线
    plt.show()

def choose_k_manually():
    # 手动选择k值
    k = int(input("Enter the number of clusters (k): "))
    return k

def fit_model(data, k):
    # 拟合KMeans模型
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters


def analyze_clusters(data, clusters):
    # 分析聚类结果
    for i in range(len(set(clusters))):
        print("Cluster", i, "Top terms:")
        cluster_data = data[clusters == i]

        # 计算词频统计
        word_freq = cluster_data.mean().sort_values(ascending=False)
        print(word_freq.head())

        # 这里你可以添加任何其他的特定分析

def main():

    # 绘制肘部法则图
    plot_elbow_method(X_train)

    # 手动选择k值
    k = choose_k_manually()
    print("Selected k:", k)

    # 拟合KMeans模型
    clusters = fit_model(X_train, k)

    # 分析聚类结果
    analyze_clusters(X_train, clusters)

if __name__ == "__main__":
    main()
