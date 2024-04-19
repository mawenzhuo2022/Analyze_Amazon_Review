# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/19 10:29
# @Function:

import os
import pandas as pd
from openai import OpenAI  # Import the OpenAI library
from dotenv import load_dotenv  # Import the function to load environment variables
import re


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
                {"role": "system", "content": f"Describe the impact of each listed feature on customer satisfaction and product ratings, using a clear, numbered format. Avoid any mention of numerical data or coefficients. Focus solely on how each feature, when present in a product, typically influences customer perceptions and reviews. Ensure each feature is discussed within the context of the specific product [[[{product}]]], mention the product name at top of feedback between [[[]]], highlighting the practical implications of each feature on user experience."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # You can adjust the temperature if you need more creative or conservative responses
        )
        # Extracting the message content from the response
        # Assuming the response structure is correctly returned in the expected format
        message_content = response.choices[0].message.content  # Accessing message content directly as provided
        print(message_content)
        return message_content.strip()
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def result_process(result):
    # Extract the product name using regex from the header
    product_name_match = re.search(r'\[\[\[(.*?)\]\]\]', result)
    if product_name_match:
        product_name = product_name_match.group(1)
        product_name_output = f"Product Name: {product_name}\n"  # Prepare product name for output
    else:
        product_name_output = ""

    lines = result.split('\n')
    max_feature_number = len(lines)
    i = 1  # Start from the second line (index 1)
    output = []  # Use a list to accumulate output lines
    all_features = {}

    while i < len(lines):
        line = lines[i].strip()
        if any(line.startswith(f"{n}.") for n in range(1, max_feature_number + 1)):
            feature = line.split('"')[1]  # Extract the feature name
            i += 1  # Move to the description line
            description = ""
            while i < len(lines) and not any(lines[i].strip().startswith(f"{n}.") for n in range(1, max_feature_number + 1)):
                description += lines[i].strip() + " "
                i += 1
            parts = description.strip().rsplit('.', -1)  # Split all sentences on period
            all_features[feature] = parts
        else:
            i += 1

    # Extract second-to-last sentence of each feature
    for feature, sentences in all_features.items():
        if len(sentences) >= 2:
            second_last_content = sentences[-2].strip() + '.'
        else:
            second_last_content = sentences[0].strip() if sentences else ""

        # Special handling for "Charge" to append additional info
        if feature == "Charge":
            second_last_content += " However, concerns related to \"battery\" and \"charge\" may slightly lower product ratings."

        output.append(f"{feature}: {second_last_content}")

    final_result = product_name_output + "\n".join(output)  # Join all output lines into a single string
    print(final_result)  # Optionally print the final result
    return final_result

    final_result = product_name_output + "\n".join(output)  # Join all output lines into a single string
    print(final_result)  # Optionally print the final result
    return final_result


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

    result = result_process(interpretation)

    # Format the filename with the product name
    filename = f"../../dat/analyze/results/{product_name}.txt"

    # Write the result to the file
    with open(filename, 'w') as file:
        file.write(result)



if __name__ == "__main__":
    main()
