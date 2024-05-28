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
                {"role": "system", "content": f"Describe the impact of each listed feature on customer satisfaction and product ratings, using a clear, numbered format. Avoid any mention of numerical data or coefficients. Focus solely on how each feature, when present in a product, typically influences customer perceptions and reviews. Ensure each feature is discussed within the context of the specific product [[[{product}]]], mention the product name at top of feedback between [[[]]], highlighting the practical implications of each feature on user experience. Sample output:'1. Feature: best, Coefficient: 0.494261\n- Having the 'best' feature in the product tends to significantly boost customer satisfaction and product ratings. Customers perceive products with this feature as top-tier and superior, leading to positive reviews and high ratings.'"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5  # You can adjust the temperature if you need more creative or conservative responses
        )
        # Extracting the message content from the response
        # Assuming the response structure is correctly returned in the expected format
        message_content = response.choices[0].message.content  # Accessing message content directly as provided
        print(message_content)
        return message_content.strip()
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def extract_last_sentence(description):
    # 提取最后一句话，假设已经正确获得了描述的最后一行
    sentences = description.rsplit('. ', 1)
    return sentences[-1].strip() + '.' if sentences[-1][-1] != '.' else sentences[-1].strip()


def result_process(result):
    # 按行分割结果文本以逐行处理
    lines = result.split('\n')

    product_name_match = re.search(r'\[\[\[(.*?)\]\]\]', result)
    product_output = f"Product Name: {product_name_match.group(1)}\n" if product_name_match else ""

    # 初始化最终输出列表
    features_output = []

    # 提取每个特征的名称和最后一句话
    for i, line in enumerate(lines):
        if "Feature:" in line and i + 1 < len(lines):  # 确保下一行存在
            # 特征名是“Feature:”后面的单词
            feature = line.split('Feature: ')[1].split(',')[0].strip()
            # 下一行的内容
            next_line = lines[i + 1]
            last_sentence = extract_last_sentence(next_line)
            features_output.append(f"{feature}: {last_sentence}\n")

    # 将所有特征和最后一句话合并到一个字符串中
    return product_output + ''.join(features_output)

def main(data):
    product_name = f'{data}'

    # Path to the regression results CSV file
    file_path = 'app/backend/dat/analyze/regression_results/regression_results.csv'
    load_dotenv()  # Load environment variables from the .env file
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # Initialize the OpenAI client with the API key

    # Load the data
    regression_results = load_regression_results(file_path)

    # Generate GPT query
    prompt = generate_gpt_query(regression_results)

    # Get interpretation from GPT
    interpretation = interpret_results_with_gpt(prompt, product_name, client)

    result = result_process(interpretation)

    return result

    # # Format the filename with the product name
    # filename = f"../../dat/analyze/results/{product_name}.txt"
    #
    # # Write the result to the file
    # with open(filename, 'w') as file:
    #     file.write(result)



if __name__ == "__main__":
    main()