import os
import csv
from openai import OpenAI
from dotenv import load_dotenv

def csv_content_to_string(filepath):
    # Initialize a list to hold the formatted reviews.
    formatted_reviews = []
    # Open the CSV file using the provided file path.
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        # Create a CSV reader to parse the file.
        reader = csv.reader(csvfile)
        # Extract column titles from the first row of the CSV.
        column_titles = next(reader)
        # Determine the number of columns based on the title row.
        num_columns = len(column_titles)

        # Iterate through each row in the CSV file, starting with the second row.
        for row_number, row in enumerate(reader, start=2):
            # Stop adding reviews if we've reached 80 reviews.
            if len(formatted_reviews) >= 80:
                break
            # Check if the row has the correct number of columns.
            if len(row) != num_columns:
                # Raise an error if the row's column count doesn't match the header's.
                raise ValueError(f"CSV format error at line {row_number}.")
            else:
                # Extract product name and review text from the row.
                product_name = row[0]
                review_text = row[1]
                # Format the review with column titles and append it to the list.
                formatted_review = f"{column_titles[0]}: {product_name}\n{column_titles[1]}: {review_text}"
                formatted_reviews.append(formatted_review)
    # Join all formatted reviews into a single string with newlines and return it.
    return "\n".join(formatted_reviews)

def read_file_content(filepath):
    # Open and read the entire content of a file specified by the filepath.
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def process_results(text):
    # Extract the product name surrounded by '{{' and '}}'.
    product_name_start = text.find('{{') + 2
    product_name_end = text.find('}}', product_name_start)
    product_name = text[product_name_start:product_name_end].strip()

    # Initialize a list to store titles of different aspects mentioned in the comments.
    aspect_titles = []
    # Split the text into parts by double newlines.
    parts = text.split('\n\n')
    for part in parts:
        # Identify parts that contain aspect titles marked by '**'.
        if '**' in part:
            aspect_start = part.find('**') + 2
            aspect_end = part.find('**', aspect_start)
            aspect_title = part[aspect_start:aspect_end]
            aspect_titles.append(aspect_title)

    # Convert the list of aspect titles to a string with each title on a new line.
    aspects_str = '\n'.join([f"- {title}" for title in aspect_titles])

    # Assemble the final output string with the product name and aspect titles.
    output = f"Product Name: {product_name}\nAspects comment mentions:\n{aspects_str}"
    return output

if __name__ == "__main__":
    # Define file paths for the CSV containing reviews and the system prompt.
    filepath = "../../dat/prompt1/data_jeans.csv"
    system_prompt_path = "../../dat/prompt1/system_prompt1.csv"

    # Convert the content of the reviews CSV file to a formatted string.
    reviews_string = csv_content_to_string(filepath)
    # Read the system prompt content from its file.
    system_prompt_content = read_file_content(system_prompt_path)

    # Load environment variables, including the OpenAI API key.
    load_dotenv()
    # Initialize the OpenAI client with the API key.
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    try:
        # Make a request to the OpenAI chat API with the formatted reviews and system prompt.
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": reviews_string}
            ]
        )
        # Process and print the structured response to extract product name and aspect titles.
        print(f"Response:\n {process_results(response.choices[0].message.content)}")
    except Exception as e:
        # Print any errors encountered during the API request or processing.
        print(f"Error: {e}")
