import os
import csv
from openai import OpenAI
from dotenv import load_dotenv






def csv_content_to_string(filepath):
    formatted_reviews = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # 读取并存储列标题
        column_titles = next(reader)
        num_columns = len(column_titles)

        for row_number, row in enumerate(reader, start=2):  # 从文件的第二行开始读取，计数从2开始
            if len(row) != num_columns:
                raise ValueError(
                    f"CSV format does not meet the requirements, the number of columns does not match the header's requirements at line {row_number}.")
            else:
                product_name = row[0]
                review_text = row[1]
                formatted_review = f"{column_titles[0]}: {product_name}\n{column_titles[1]}: {review_text}"
                formatted_reviews.append(formatted_review)

    return formatted_reviews


if __name__ == "__main__":
    # filepath
    filepath = "../../dat/prompt1/data_jeans.csv"
    # system prompt path
    system_prompt_path = "../../dat/prompt1/system_prompt1.csv"

    # testcase ########################################
    reviews = csv_content_to_string(filepath)
    for review in reviews[:5]:  # 打印前5条评论作为示例
        print(review)
        print("-----")
    ###################################################

    csvcontent = csv_content_to_string(filepath)

    # load environment variable 
    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # get the data
    data_path = 'data.csv'
    prompt = f"Here is a list of Amazon reviews about Product Name. Please summarize different aspects covered in it :"

    # Send the prompt to the API
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt_path},
                {"role": "user", "content": csvcontent}
            ]
        )

        print(f"Response: {response.choices[0].message.content}")


    except Exception as e:
        # If there was an error, print the error message
        print(f"Error: {e}")






