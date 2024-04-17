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
    return pd.read_csv(filepath)

def create_cluster_summaries(data):
    cluster_groups = data.groupby('Cluster')
    summaries = {}
    for cluster, group in cluster_groups:
        top_features = group.drop('Cluster', axis=1).mean().sort_values(ascending=False).head(5).index.tolist()
        summaries[cluster] = " ".join(top_features)
    return summaries


def identify_keywords(summaries, client):
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
        # Extract the keyword text from the response
        keywords_text = response.choices[0].message.content

        # Normalize the text by converting it to lowercase to handle case insensitivity
        keywords_text = keywords_text.lower().replace("key words and phrases:", "").strip()

        # Replace newlines with commas and split the string into a list
        split_keywords = keywords_text.replace("\n", ",").split(",")

        # Clean each keyword to remove unwanted characters and whitespace
        cleaned_keywords = [kw.strip(' ,.-') for kw in split_keywords if kw.strip(' ,.-')]

        # Store the cleaned keywords in the dictionary
        keywords[cluster] = cleaned_keywords

    return keywords


def main():
    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    filepath = "../../dat/analyze/cluster/cluster.csv"
    data = load_clustered_data(filepath)
    cluster_summaries = create_cluster_summaries(data)
    cluster_keywords = identify_keywords(cluster_summaries, client)

    # Create a DataFrame from the dictionary of keywords
    keywords_df = pd.DataFrame(list(cluster_keywords.items()), columns=['Cluster', 'Keywords'])

    # Save the DataFrame to a CSV file
    keywords_df.to_csv('../../dat/analyze/feature_keywords/feature_keywords.csv', index=False)

    for cluster, ks in cluster_keywords.items():
        print(f"Cluster {cluster} Keywords: {ks}")

if __name__ == "__main__":
    main()