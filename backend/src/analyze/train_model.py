# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/17 11:21
# @Function: Train a linear regression model and evaluate its performance
# @功能：训练一个线性回归模型并评估其性能

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def load_keywords(keyword_path):
    # Load keywords from a CSV file
    # 从CSV文件加载关键词
    keywords_df = pd.read_csv(keyword_path)
    keywords_df['Keywords'] = keywords_df['Keywords'].apply(eval)
    keywords = set(sum(keywords_df['Keywords'].tolist(), []))
    return keywords

def filter_features(X, keywords):
    # Filter the feature set to keep only those columns that are in the keyword list
    # 过滤特征集，仅保留关键词列表中的列
    filtered_columns = [col for col in X.columns if col in keywords]
    return X[filtered_columns]

def load_and_filter_data(features_path, target_path, keywords):
    # Load feature and target datasets and apply the keyword filter
    # 加载特征和目标数据集并应用关键词过滤
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).squeeze()
    X_filtered = filter_features(X, keywords)
    return X_filtered, y

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Setting up the Ridge regression model
    # 设置岭回归模型
    model = Ridge()
    param_grid = {'alpha': [0.1, 1, 10, 100, 1000]}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)

    # Train the model using Grid Search
    # 使用网格搜索训练模型
    grid_search.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
    best_model = grid_search.best_estimator_
    print(f"Best parameters for Ridge: ", grid_search.best_params_)

    # Make predictions and evaluate the model
    # 进行预测并评估模型
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE for Ridge: {mse}")

    # Extract coefficients and feature ranking
    # 提取系数和特征排名
    if hasattr(best_model, 'coef_'):
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': best_model.coef_
        }).sort_values(by='Coefficient', key=abs, ascending=False)
        print("Feature ranking for Ridge:")
        print(feature_importance)

        # Save the feature importance to a CSV file
        # 将特征重要性保存到CSV文件
        feature_importance.to_csv('../../dat/analyze/regression_results/regression_results.csv', index=False)

def main():
    # Load keywords and filtered data, then train and evaluate the model
    # 加载关键词和过滤后的数据，然后训练和评估模型
    keywords = load_keywords('../../dat/analyze/feature_keywords/feature_keywords.csv')
    X_train, y_train = load_and_filter_data('../../dat/analyze/cleaned_data/X_train.csv',
                                            '../../dat/analyze/cleaned_data/Y_train.csv', keywords)
    X_test, y_test = load_and_filter_data('../../dat/analyze/cleaned_data/X_test.csv',
                                          '../../dat/analyze/cleaned_data/Y_test.csv', keywords)
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
