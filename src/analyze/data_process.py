# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/15 10:16
# @Function: Text processing utilities

import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet

# Download all resources recommended by NLTK, this may take some time
nltk.download('popular')

# Download specific resources individually or re-download them
nltk.download('stopwords')  # Download stopwords
nltk.download('wordnet')  # Download WordNet
nltk.download('averaged_perceptron_tagger')  # Download the averaged perceptron tagger
nltk.download('punkt')  # Download the Punkt tokenizer

# Initialize a list to hold the formatted reviews.
# 初始化一个列表以保存格式化后的评论。

stop_words = set(stopwords.words('english'))  # Initialize a list of stopwords
# Initialize a lemmatizer object to reduce words to their base form.
# 初始化一个词形还原器对象，将单词转换为它们的基本形式。
lemmatizer = WordNetLemmatizer()

# Function to convert CSV content to a string
# 将CSV内容转换为字符串的函数
def csv_content_to_string(filepath):
    formatted_reviews = []  # Initialize an empty list to store the formatted reviews
    # Open the CSV file using the provided file path.
    # 使用提供的文件路径打开CSV文件。
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)  # Create a CSV reader object
        column_titles = next(reader)  # Extract column titles from the first row of the CSV
        num_columns = len(column_titles)  # Determine the number of columns based on the title row

        # Iterate through each row in the CSV file.
        for row_number, row in enumerate(reader, start=2):
            if len(formatted_reviews) >= 80:  # Stop adding reviews if we've reached 80 reviews
                break
            if len(row) != num_columns:  # Check if the row has the correct number of columns
                raise ValueError(f"CSV format error at line {row_number}.")  # Raise an error if the column count doesn't match
            else:
                # Extract product name, rating, and review text from the row
                product_name = row[0]
                rating = row[1]
                review_text = row[2]
                # Format the review with column titles and append it to the list
                formatted_review = f"{column_titles[0]}: {product_name}\n{column_titles[1]}: {rating}\n{column_titles[2]}: {review_text}"
                formatted_reviews.append(formatted_review)
    return "\n".join(formatted_reviews)  # Join all formatted reviews into a single string with newlines and return it

# Function to map POS tag to first character lemmatize() accepts
# 将POS标签映射到词形还原器接受的第一个字符的函数
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()  # Get the POS tag of the word and convert it to uppercase
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}  # Dictionary mapping POS tags to WordNet tags
    return tag_dict.get(tag, wordnet.NOUN)  # Return the corresponding WordNet tag if found, else return the default noun tag

# Function to clean text by removing punctuation, converting to lowercase, tokenizing, removing stopwords, and lemmatizing using POS tags
# 清理文本的函数，通过去除标点符号、转换为小写、分词、去除停用词和使用POS标签进行词形还原
def clean_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    words = nltk.word_tokenize(text)  # Tokenize the text into words
    cleaned_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words if word not in stop_words]  # Lemmatize words and remove stopwords
    return cleaned_words  # Return the cleaned words

# Function to get sentiment polarity using TextBlob
# 使用TextBlob获取情感极性的函数
def get_sentiment(text):
    blob = TextBlob(text)  # Create a TextBlob object
    return blob.sentiment.polarity  # Return the polarity of the text

# Function to perform feature extraction using TF-IDF
# 使用TF-IDF进行特征提取的函数
def feature_extraction(texts):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Initialize a TF-IDF vectorizer
    features = tfidf_vectorizer.fit_transform(texts)  # Fit and transform the texts
    features_df = pd.DataFrame(features.todense(), columns=tfidf_vectorizer.get_feature_names_out())  # Convert to DataFrame
    return features_df  # Return the DataFrame of features

# Main function
# 主函数
def main():
    filepath = "../../dat/analyze/dataset_phone.csv"  # Path to the CSV file
    try:
        reviews_string = csv_content_to_string(filepath)  # Convert CSV content to string
        cleaned_reviews = clean_text(reviews_string)  # Clean the reviews

        features_df = feature_extraction(cleaned_reviews)  # Perform feature extraction
        sentiments = [get_sentiment(review) for review in cleaned_reviews]  # Calculate sentiment scores
        print("sentiments")  # Print header for sentiment scores
        print(sentiments)  # Print sentiment scores

        features_df['sentiment'] = sentiments  # Add sentiment scores as a feature
        print(features_df['sentiment'])  # Print the sentiment column

        X_train, X_test, y_train, y_test = train_test_split(features_df.drop('sentiment', axis=1),  # Split the data into train and test sets
                                                            features_df['sentiment'], test_size=0.2, random_state=42)

        print("Train and test data prepared.")  # Print message indicating data preparation completion
        print("训练数据集 X_train 的前几行：")  # Print message indicating display of first few rows of training data
        print(X_train.head())  # Display first few rows of X_train

        print("\n训练数据集 y_train 的前几行：")  # Print message indicating display of first few rows of training labels
        print(y_train.head())  # Display first few rows of y_train

    except Exception as e:
        print(f"An error occurred: {e}")  # Print error message if an exception occurs

if __name__ == '__main__':
    main()  # Call the main function if the script is executed directly
