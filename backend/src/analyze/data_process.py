# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/15 10:16
# @Function: Text processing utilities

import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet
import os

# Download all resources recommended by NLTK, this may take some time
nltk.download('popular')

# Download specific resources individually or re-download them
nltk.download('stopwords')  # Download stopwords
nltk.download('wordnet')  # Download WordNet
nltk.download('averaged_perceptron_tagger')  # Download the averaged perceptron tagger
nltk.download('punkt')  # Download the Punkt tokenizer

# Initialize a list to hold the formatted reviews.
# 初始化一个列表以保存格式化后的评论。


# Initialize a lemmatizer object to reduce words to their base form.
# 初始化一个词形还原器对象，将单词转换为它们的基本形式。
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) | {'iphone', 'apple', 'phone', 'mobile', 'device', 'product'} # Initialize a list of stopwords


# Function to convert CSV content to a string
# 将CSV内容转换为字符串的函数
def csv_content_to_string(filepath):
    formatted_reviews = []  # 初始化一个列表以保存格式化后的评论
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)  # 创建一个CSV阅读器对象
        column_titles = next(reader)  # 从CSV的第一行提取列标题

        for row in reader:  # 遍历CSV文件中的每一行
            if len(formatted_reviews) >= 2000:  # 如果已达到80条评论则停止
                break
            # 将产品名称、评分和评论文本提取出来并格式化
            product_name = row[0]
            rating = row[1]
            review_text = row[2]
            formatted_review = f"{product_name}, {rating}, {review_text}"
            formatted_reviews.append(formatted_review)
    return formatted_reviews  # 返回一个包含所有格式化后的评论的列表

# Function to map POS tag to first character lemmatize() accepts
# 将POS标签映射到词形还原器接受的第一个字符的函数
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# Function to clean text by removing punctuation, converting to lowercase, tokenizing, removing stopwords, and lemmatizing using POS tags
# 清理文本的函数，通过去除标点符号、转换为小写、分词、去除停用词和使用POS标签进行词形还原
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
        text = re.sub(r'\d+', '', text)  # 删除数字
        text = re.sub(r'\W+', ' ', text)  # 替换非字母数字字符
        text = text.lower()  # 转小写
        words = nltk.word_tokenize(text)  # 分词
        cleaned_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words if word not in stop_words]  # 词形还原且去除停用词
        return ' '.join(cleaned_words)  # 返回处理后的文本字符串
    return ""

# Function to perform feature extraction using TF-IDF
# 使用TF-IDF进行特征提取的函数
def feature_extraction(texts):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=0.10, max_df=0.95, stop_words='english')
    features = tfidf_vectorizer.fit_transform(texts)
    features_df = pd.DataFrame(features.todense(), columns=tfidf_vectorizer.get_feature_names_out())
    return features_df

# Function to get sentiment polarity using TextBlob
# 使用TextBlob获取情感极性的函数
def get_sentiment(text):
    blob = TextBlob(text)  # Create a TextBlob object
    return blob.sentiment.polarity  # Return the polarity of the text


def save_data_to_csv(data, filename, index=False):
    """
    Save a DataFrame or Series to a CSV file.

    Parameters:
    - data (DataFrame or Series): The Pandas DataFrame or Series to save.
    - filename (str): The filename for the CSV file.
    - index (bool): Whether to write row names (index). Default is False.
    """
    # Convert Series to DataFrame if necessary
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Save to CSV
    data.to_csv(filename, index=index)

# Main function
# 主函数
def main(data):
    base_path = os.path.abspath("../../dat/analyze/dataset")
    filepath = os.path.join(base_path, f"{data}.csv")
    try:
        reviews = csv_content_to_string(filepath)  # 从CSV获取内容并转换为字符串列表
        cleaned_reviews = [clean_text(review) for review in reviews]  # 清洗每条评论
        features_df = feature_extraction(cleaned_reviews)  # Perform feature extraction
        sentiments = [get_sentiment(review) for review in cleaned_reviews]  # Calculate sentiment scores
        features_df['sentiment'] = sentiments  # Add sentiment scores as a feature
        X_train, X_test, y_train, y_test = train_test_split(features_df.drop('sentiment', axis=1),  # Split the data into train and test sets
                                                            features_df['sentiment'], test_size=0.2, random_state=42)

        print("Train and test data prepared.")  # Print message indicating data preparation completion
        print("训练数据集 X_train 的前几行：")  # Print message indicating display of first few rows of training data
        print(X_train.head())  # Display first few rows of X_train

        print("\n训练数据集 y_train 的前几行：")  # Print message indicating display of first few rows of training labels
        print(y_train.head())  # Display first few rows of y_train

        save_data_to_csv(X_train, '../../dat/analyze/cleaned_data/X_train.csv')
        save_data_to_csv(X_test, '../../dat/analyze/cleaned_data/X_test.csv')
        save_data_to_csv(y_train, '../../dat/analyze/cleaned_data/Y_train.csv')
        save_data_to_csv(y_test, '../../dat/analyze/cleaned_data/Y_test.csv')


    except Exception as e:
        print(f"An error occurred: {e}")  # Print error message if an exception occurs

if __name__ == '__main__':
    main()  # Call the main function if the script is executed directly
