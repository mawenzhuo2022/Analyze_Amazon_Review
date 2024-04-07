import os
import csv
from openai import OpenAI
from dotenv import load_dotenv



# convert the csv content to string: 
def csv_content_to_string(filepath):
    content = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Join the columns with commas and append to the content list
            content.append(','.join(row))
    # Join all rows into a single string, separating rows with new lines
    return '\n'.join(content)


if __name__ == "__main__":
    csvcontent = csv_content_to_string()

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





