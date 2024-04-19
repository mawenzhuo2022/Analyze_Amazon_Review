# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/17 11:21
# @Function: Train a linear regression model and evaluate its performance

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def load_keywords(keyword_path):
    keywords_df = pd.read_csv(keyword_path)
    keywords_df['Keywords'] = keywords_df['Keywords'].apply(eval)
    keywords = set(sum(keywords_df['Keywords'].tolist(), []))
    return keywords


def filter_features(X, keywords):
    filtered_columns = [col for col in X.columns if col in keywords]
    return X[filtered_columns]


def load_and_filter_data(features_path, target_path, keywords):
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).squeeze()
    X_filtered = filter_features(X, keywords)
    return X_filtered, y


def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Setting up the Ridge regression model
    model = Ridge()
    param_grid = {'alpha': [0.1, 1, 10, 100, 1000]}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)

    grid_search.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
    best_model = grid_search.best_estimator_
    print(f"Best parameters for Ridge: ", grid_search.best_params_)

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE for Ridge: {mse}")

    # Extract coefficients and feature ranking
    if hasattr(best_model, 'coef_'):
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': best_model.coef_
        }).sort_values(by='Coefficient', key=abs, ascending=False)
        print("Feature ranking for Ridge:")
        print(feature_importance)

        # Save the feature importance to a CSV file
        feature_importance.to_csv('../../dat/analyze/regression_results/regression_results.csv', index=False)


def main():
    keywords = load_keywords('../../dat/analyze/feature_keywords/feature_keywords.csv')
    X_train, y_train = load_and_filter_data('../../dat/analyze/cleaned_data/X_train.csv',
                                            '../../dat/analyze/cleaned_data/Y_train.csv', keywords)
    X_test, y_test = load_and_filter_data('../../dat/analyze/cleaned_data/X_test.csv',
                                          '../../dat/analyze/cleaned_data/Y_test.csv', keywords)
    train_and_evaluate(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
