# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/17 9:48
# @Function: Identify keywords and phrases for each cluster using OpenAI's API.

import os
import pandas as pd
from openai import OpenAI  # Import the OpenAI library
from dotenv import load_dotenv  # Import the function to load environment variables
import numpy as np

def load_clustered_data(filepath):
    """Load clustered data from a CSV file."""
    return pd.read_csv(filepath)  # Return the loaded data as a DataFrame

def create_cluster_summaries(data):
    """Create summaries for each cluster based on the top features."""
    cluster_groups = data.groupby('Cluster')  # Group the data by clusters
    summaries = {}
    for cluster, group in cluster_groups:
        top_features = group.drop('Cluster', axis=1).mean().sort_values(ascending=False).head(5).index.tolist()
        # Calculate the mean of each feature for the cluster, sort them in descending order,
        # and select the top 5 features
        summaries[cluster] = " ".join(top_features)  # Concatenate the top features into a summary string
    return summaries  # Return a dictionary containing cluster summaries

def identify_keywords(summaries, client):
    """Identify keywords and phrases for each cluster using OpenAI's API."""
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

    return keywords  # Return a dictionary containing keywords for each cluster

def main():
    load_dotenv()  # Load environment variables from the .env file
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # Initialize the OpenAI client with the API key

    filepath = "app/backend/dat/analyze/cluster/cluster.csv"  # Define the filepath for the clustered data
    data = load_clustered_data(filepath)  # Load the clustered data
    cluster_summaries = create_cluster_summaries(data)  # Create summaries for each cluster
    cluster_keywords = identify_keywords(cluster_summaries, client)  # Identify keywords for each cluster

    # Create a DataFrame from the dictionary of keywords
    keywords_df = pd.DataFrame(list(cluster_keywords.items()), columns=['Cluster', 'Keywords'])

    # Save the DataFrame to a CSV file
    keywords_df.to_csv('app/backend/dat/analyze/feature_keywords/feature_keywords.csv', index=False)

    # Print the keywords for each cluster
    for cluster, ks in cluster_keywords.items():
        print(f"Cluster {cluster} Keywords: {ks}")

if __name__ == "__main__":
    main()
