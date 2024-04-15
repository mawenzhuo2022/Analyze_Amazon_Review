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

# 下载NLTK推荐的所有资源，这可能需要一些时间
nltk.download('popular')

# 单独下载或重新下载特定资源
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


# 初始化停用词列表和词形还原器
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


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

        # Iterate through each row in the CSV file.
        for row_number, row in enumerate(reader, start=2):
            # Stop adding reviews if we've reached 80 reviews.
            if len(formatted_reviews) >= 80:
                break
            # Check if the row has the correct number of columns.
            if len(row) != num_columns:
                # Raise an error if the row's column count doesn't match the header's.
                raise ValueError(f"CSV format error at line {row_number}.")
            else:
                # Extract product name, rating and review text from the row.
                product_name = row[0]
                rating = row[1]
                review_text = row[2]
                # Format the review with column titles and append it to the list.
                formatted_review = f"{column_titles[0]}: {product_name}\n{column_titles[1]}: {rating}\n{column_titles[2]}: {review_text}"
                formatted_reviews.append(formatted_review)
    # Join all formatted reviews into a single string with newlines and return it.
    return "\n".join(formatted_reviews)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def clean_text(text):
    # 去除标点符号
    text = ''.join([char for char in text if char not in string.punctuation])
    # 转换为小写
    text = text.lower()
    # 分词
    words = nltk.word_tokenize(text)
    # 去除停用词并进行词形还原，使用POS标签
    cleaned_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words if word not in stop_words]
    return cleaned_words

def get_sentiment(text):
    # 情感分析得分计算
    blob = TextBlob(text)
    return blob.sentiment.polarity

def feature_extraction(texts):
    # 使用TF-IDF进行特征提取
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    features = tfidf_vectorizer.fit_transform(texts)
    # 转换为DataFrame
    features_df = pd.DataFrame(features.todense(), columns=tfidf_vectorizer.get_feature_names_out())
    return features_df


def main():
    filepath = "../../dat/analyze/dataset_phone.csv"
    try:
        reviews_string = csv_content_to_string(filepath)
        cleaned_reviews = clean_text(reviews_string)

        # 特征提取
        features_df = feature_extraction(cleaned_reviews)
        # 计算情感分析得分
        sentiments = [get_sentiment(review) for review in cleaned_reviews]

        # 将情感分析得分添加为一个特征
        features_df['sentiment'] = sentiments

        # 数据划分
        X_train, X_test, y_train, y_test = train_test_split(features_df.drop('sentiment', axis=1),
                                                            features_df['sentiment'], test_size=0.2, random_state=42)

        print("Train and test data prepared.")
        print("训练数据集 X_train 的前几行：")
        print(X_train.head())

        print("\n训练数据集 y_train 的前几行：")
        print(y_train.head())


    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
