import os
import csv
from openai import OpenAI
from dotenv import load_dotenv






def csv_content_to_string(filepath):
    formatted_reviews = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            product_name = row[0].strip('"')
            review_text = row[1].replace('\n', ' ')  # Replace newlines in review text, if any
            formatted_review = f"Product name: {product_name}\nProduct review: {review_text}"
            formatted_reviews.append(formatted_review)
    return formatted_reviews


if __name__ == "__main__":
    # filepath
    filepath = "../../dat/prompt1/data_jeans.csv"

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
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        print(f"Response: {response.choices[0].message.content}")


    except Exception as e:
        # If there was an error, print the error message
        print(f"Error: {e}")






