# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/19 10:29
# @Function:

import os
import pandas as pd
from openai import OpenAI  # Import the OpenAI library
from dotenv import load_dotenv  # Import the function to load environment variables


def load_regression_results(file_path):
    # Load the regression results from a CSV file
    return pd.read_csv(file_path)


def generate_gpt_query(df):
    """
    Create a text prompt for the GPT model to interpret the regression results.
    This function constructs a prompt that details each feature with its corresponding
    coefficient and asks the GPT to explain the impact of these features on product ratings.
    """
    # Start with an introduction explaining what needs to be done.
    prompt = "Interpret the following regression coefficients for a product rating model and explain how each feature impacts the product's ratings:\n\n"

    # Iterate over each row in the DataFrame to add feature details to the prompt.
    for index, row in df.iterrows():
        prompt += f"Feature: {row['Feature']}, Coefficient: {row['Coefficient']:.6f}\n"

    # Append a concluding line to request a detailed explanation of significance.
    prompt += "\nExplain the significance of these features based on their coefficients."

    return prompt


def interpret_results_with_gpt(prompt, product, client):
    """Function to call the GPT API and get an interpretation using the chat interface."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Adjust the model identifier as necessary
            messages=[
                {"role": "system", "content": f"Interpret the significance of the following features based on their coefficients for a product rating model. This is the dataset for [[[{product}]]]. Place the product name between [[[]]] in introduction."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2  # You can adjust the temperature if you need more creative or conservative responses
        )
        # Extracting the message content from the response
        # Assuming the response structure is correctly returned in the expected format
        message_content = response.choices[0].message.content  # Accessing message content directly as provided
        return message_content.strip()
    except Exception as e:
        print(f"Error occurred: {e}")
        return None




def main():
    product_name = 'iphone7 (refurbrished)'

    # Path to the regression results CSV file
    file_path = '../../dat/analyze/regression_results/regression_results.csv'
    load_dotenv()  # Load environment variables from the .env file
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # Initialize the OpenAI client with the API key

    # Load the data
    regression_results = load_regression_results(file_path)

    # Generate GPT query
    prompt = generate_gpt_query(regression_results)

    # Get interpretation from GPT
    interpretation = interpret_results_with_gpt(prompt, product_name, client)

    print(interpretation)


if __name__ == "__main__":
    main()
