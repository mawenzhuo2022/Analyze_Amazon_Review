# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/17 11:21
# @Function:

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler  # 导入 StandardScaler 用于特征缩放

def load_keywords(keyword_path):
    # 加载关键词并将字符串表示的列表转换为实际列表
    # Load keywords and convert string representation of list to actual list
    keywords_df = pd.read_csv(keyword_path)
    keywords_df['Keywords'] = keywords_df['Keywords'].apply(eval)
    # 将列表的列表扁平化成一个单一的关键词列表
    # Flatten the list of lists into a single list of keywords
    keywords = set(sum(keywords_df['Keywords'].tolist(), []))
    return keywords

def filter_features(X, keywords):
    # 根据关键词集过滤 X 中的列
    # Filter columns in X based on whether they are in the keywords set
    filtered_columns = [col for col in X.columns if col in keywords]
    return X[filtered_columns]

def load_and_filter_data(features_path, target_path, keywords):
    # 加载数据并根据关键词过滤
    # Load data and apply filtering based on keywords
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).squeeze()  # 将 y 转换为 Series
    X_filtered = filter_features(X, keywords)
    return X_filtered, y

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # 可选：归一化数据（取消下两行的注释以应用归一化）
    # Optional: Normalize the data (uncomment the next two lines to apply normalization)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)

    # 训练线性回归模型并评估其性能
    # Train a linear regression model and evaluate its performance
    model = LinearRegression(fit_intercept=True, n_jobs=-1)  # 根据需要调整 fit_intercept 和 n_jobs
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 计算并打印 RMSE 和 MAE
    # Calculate and print RMSE and MAE
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")

    # 执行交叉验证并计算 MSE
    # Perform cross-validation and calculate MSE
    cv_scores = cross_val_score(model, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), cv=5,
                                scoring='neg_mean_squared_error')
    print(f"Cross-validated MSE: {-cv_scores.mean()}")

def main():
    # 加载关键词
    # Load keywords
    keywords = load_keywords('../../dat/analyze/feature_keywords/feature_keywords.csv')

    # 加载和过滤数据基于关键词
    # Load and filter data based on keywords
    X_train, y_train = load_and_filter_data('../../dat/analyze/cleaned_data/X_train.csv',
                                            '../../dat/analyze/cleaned_data/Y_train.csv', keywords)
    X_test, y_test = load_and_filter_data('../../dat/analyze/cleaned_data/X_test.csv',
                                          '../../dat/analyze/cleaned_data/Y_test.csv', keywords)

    # 训练模型并评估其性能
    # Train the model and evaluate its performance
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
