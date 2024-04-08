import os
import csv
from openai import OpenAI
from dotenv import load_dotenv

def csv_content_to_string(filepath):
    formatted_reviews = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        column_titles = next(reader)
        num_columns = len(column_titles)

        for row_number, row in enumerate(reader, start=2):
            if len(formatted_reviews) >= 80:
                break
            if len(row) != num_columns:
                raise ValueError(f"CSV format error at line {row_number}.")
            else:
                product_name = row[0]
                review_text = row[1]
                formatted_review = f"{column_titles[0]}: {product_name}\n{column_titles[1]}: {review_text}"
                formatted_reviews.append(formatted_review)
    return "\n".join(formatted_reviews)

def read_file_content(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


def process_results(text):
    # 提取产品名称，修正以匹配完整的产品名称
    product_name_start = text.find('{{') + 2
    product_name_end = text.find('}}', product_name_start)
    product_name = text[product_name_start:product_name_end].strip()


    # 查找并提取所有评论方面的标题
    aspect_titles = []
    parts = text.split('\n\n')
    for part in parts:
        if '**' in part:  # 查找包含标题的部分
            aspect_start = part.find('**') + 2
            aspect_end = part.find('**', aspect_start)
            aspect_title = part[aspect_start:aspect_end]
            aspect_titles.append(aspect_title)

    # 将标题列表转换为字符串
    aspects_str = '\n'.join([f"- {title}" for title in aspect_titles])

    # 组装最终输出
    output = f"Product Name: {product_name}\nAspects comment mentions:\n{aspects_str}"
    return output

if __name__ == "__main__":
    filepath = "../../dat/prompt1/data_jeans.csv"
    system_prompt_path = "../../dat/prompt1/system_prompt1.csv"

    reviews_string = csv_content_to_string(filepath)
    system_prompt_content = read_file_content(system_prompt_path)

    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": reviews_string}
            ]
        )
        print(f"Response:\n {process_results(response.choices[0].message.content)}")
        #print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")
