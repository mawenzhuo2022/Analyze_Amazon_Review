# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/17 11:21
# @Function: Train a linear regression model and evaluate its performance

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(max_iter=10000),
        'ElasticNet': ElasticNet(max_iter=10000)
    }
    param_grid = {
        'Ridge': {'alpha': [0.1, 1, 10, 100, 1000]},
        'Lasso': {'alpha': [0.01, 0.1, 1, 10, 100]},
        'ElasticNet': {'alpha': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.2, 0.5, 0.8]}
    }

    for name, model in models.items():
        print(f"\nTuning {name} model")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid[name], cv=5, scoring='neg_mean_squared_error',
                                   verbose=1)
        grid_search.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

        print(f"Best parameters for {name}: ", grid_search.best_params_)
        print(f"Best score for {name} (negative MSE): ", grid_search.best_score_)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Test MSE for {name}: {mse}")

    #model = Ridge()
    #grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
    #                           verbose=1)

    #grid_search.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
    #print("Best parameters:", grid_search.best_params_)
    #print("Best score (negative MSE):", grid_search.best_score_)

    #best_model = grid_search.best_estimator_
    #y_pred = best_model.predict(X_test)
    #mse = mean_squared_error(y_test, y_pred)
    #print(f"Test MSE: {mse}")
    #model.fit(X_train, y_train)
    #y_pred = model.predict(X_test)

    # 计算并打印 RMSE 和 MAE
    # Calculate and print RMSE and MAE
    #rmse = root_mean_squared_error(y_test, y_pred)
    #mae = mean_absolute_error(y_test, y_pred)
    #print(f"Fit Intercept: {fit_intercept}, n_jobs: {n_jobs}, RMSE: {rmse}, MAE: {mae}")

    # 执行交叉验证并计算 MSE
    # Perform cross-validation and calculate MSE
    #cv_scores = cross_val_score(model, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), cv=5,
    #                            scoring='neg_mean_squared_error')
    #print(f"Cross-validated MSE: {-cv_scores.mean()}")

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
    #for fit_intercept in [True, False]:
    #    for n_jobs in [-1, 1]:
    #        train_and_evaluate(X_train, X_test, y_train, y_test, fit_intercept, n_jobs)
    #Fit Intercept = True, n_jobs = -1
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
