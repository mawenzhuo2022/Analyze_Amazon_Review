# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/17 9:48
# @Function:

import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

def load_clustered_data(filepath):
    # 加载数据
    data = pd.read_csv(filepath)
    return data

def create_cluster_summaries(data, text_column):
    # 为每个聚类创建文本描述
    cluster_groups = data.groupby('Cluster')
    summaries = {}
    for cluster, group in cluster_groups:
        # 假设每个聚类的描述是基于一个特定的文本列生成的
        top_features = group.mean().sort_values(ascending=False).head(5).index.tolist()
        summaries[cluster] = " ".join(top_features)
    return summaries

def identify_keywords(summaries, client):
    # 向 GPT API 发送聚类描述，请求提取关键词
    keywords = {}
    for cluster, text in summaries.items():
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Identify key words and phrases from the following cluster description:"},
                {"role": "user", "content": text}
            ],
            max_tokens=50,
            temperature=0.5
        )
        keywords_text = response.choices[0].message.content
        cleaned_keywords = keywords_text.replace("Key words and phrases:", "").replace("-", "").replace(",", "").split()
        cleaned_keywords = [kw.strip(' ,.-') for kw in cleaned_keywords if kw.strip(' ,.-')]
        keywords[cluster] = cleaned_keywords
    return keywords

def main():
    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    filepath = "../../dat/analyze/cluster/cluster.csv"
    data = load_clustered_data(filepath)
    # 假设实际的文本列名是 'review'
    cluster_summaries = create_cluster_summaries(data, 'review')
    cluster_keywords = identify_keywords(cluster_summaries, client)

    for cluster, ks in cluster_keywords.items():
        print(f"Cluster {cluster} Keywords: {ks}")

if __name__ == "__main__":
    main()