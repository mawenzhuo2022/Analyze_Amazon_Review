[//]: # (# -*- coding: utf-8 -*-)

[//]: # (# @Author  : Wenzhuo Ma)

[//]: # (# @Time    : 2024/4/13)

# 产品评论分析与评分预测
# Product Review Analysis and Rating Prediction

本项目旨在使用 Bag-of-Words (BoW) 模型和线性回归方法对产品评论进行分析，以预测和解释用户的评分。此方法通过聚类分析和回归分析来量化不同方面的评论内容对产品评分的影响。
This project aims to analyze product reviews using the Bag-of-Words (BoW) model and linear regression methods to predict and explain user ratings. This method quantifies the impact of different aspects of review content on product ratings through cluster analysis and regression analysis.

# **Important:** INSTALL INSTRUCTIONS

To install the required packages, run the following commands:
````
pip install textblob
pip install nltk
pip install scikit-learn
````

## 步骤概述
## Step Overview

### 步骤 1: 数据准备和预处理
### Step 1: Data Preparation and Preprocessing
- **收集数据**：获取包含产品评论和相关评分的数据集。
  - **Collect Data**: Acquire a dataset containing product reviews and corresponding ratings.
- **文本清洗**：去除文本中的无用符号和停用词，并进行词干提取或词形还原。
  - **Text Cleaning**: Remove useless symbols and stopwords from the text, and perform stemming or lemmatization.
- **标签化**：将清洗后的文本转换为 BoW 向量，可选择加入 N-grams 以捕捉词序信息。
  - **Tokenization**: Convert the cleaned text into BoW vectors, optionally incorporating N-grams to capture word order information.

### 步骤 2: 聚类分析
### Step 2: Cluster Analysis
- **选择聚类算法**：使用 K-means 算法对 BoW 向量进行聚类。
  - **Choose Clustering Algorithm**: Use K-means algorithm to cluster BoW vectors.
- **确定聚类数**：通过肘部方法等技术确定聚类的最佳数量。
  - **Determine Number of Clusters**: Determine the optimal number of clusters using techniques like the Elbow Method.
- **执行聚类**：将评论进行聚类，每个聚类代表相似的评论集合。
  - **Perform Clustering**: Cluster the reviews, with each cluster representing a similar set of reviews.

### 步骤 3: 使用 GPT API 进行关键词识别
### Step 3: Keyword Identification Using GPT API
- **准备聚类描述**：为每个聚类生成文本描述，选择最具代表性的特征。这可以基于统计量，如聚类内特征的平均值或中位数。
  - **Prepare Cluster Descriptions**: Generate a text description for each cluster by selecting the most representative features. This can be based on statistical measures such as the mean or median values of the features within the cluster.
- **构建 API 请求**：使用 GPT-3.5 或更新版本构建请求，包括聚类描述。请求应指定需要识别关键词或关键短语，系统提示指导模型分析内容，例如“从以下聚类描述中识别关键字和短语”。
  - **Build API Requests**: Configure a request using GPT-3.5 or later, including the cluster description. The request should specify the need to identify keywords or key phrases, with a system prompt guiding the model on what to analyze, e.g., "Identify key words and phrases from the following cluster description."
- **发送请求和处理响应**：将准备好的文本作为 API 请求的一部分发送。调整参数，如`温度`以控制创造性或`最大标记`以限制响应长度。
  - **Send Request and Process Responses**: Send the prepared text as part of the API request. Adjust parameters like `temperature` to control creativity or `max_tokens` to limit response length.
- **分析和应用关键词**：评估这些关键词在聚类背景中的重要性和相关性。探索这些关键词与用户评分、客户满意度或其他业务指标的相关性。
  - **Analyze and Apply Keywords**: Evaluate the importance and relevance of these keywords within the cluster context. Explore how these keywords correlate with user ratings, customer satisfaction, or other business metrics.

### 步骤 4: 训练和评估模型
### Step 4: Training and Evaluating the Model

- **模型训练**：准备数据，确保所有输入特征包括新识别的关键词已准备好进行建模。检查数据完整性，并适当处理任何缺失或异常值。使用线性回归模型，以增强的特征向量作为输入，以用户评分作为响应变量进行模型训练。
  - **Model Training**: Prepare the data ensuring all input features, including the newly identified keywords, are ready for modeling. Check for data integrity and handle any missing or outlier values appropriately. Train the model using a linear regression model, with enriched feature vectors as inputs and user ratings as the response variable.

- **模型评估**：使用交叉验证等技术评估模型的性能和泛化能力。这有助于评估预测模型的准确性和可靠性。利用指标如均方根误差（RMSE）和平均绝对误差（MAE）等来量化模型在评估阶段的性能。
  - **Model Evaluation**: Evaluate the model's performance and generalization ability using techniques like cross-validation. This helps in assessing the accuracy and reliability of the predictive model. Utilize metrics such as Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) to quantify the model's performance during the evaluation phase.


### 步骤 5: 量化贡献
### Step 5: Quantifying Contributions
- **分析回归系数**：解释线性回归模型中的系数，以确定每个特征的影响力。
  - **Analyze Regression Coefficients**: Interpret the coefficients in the linear regression model to determine the impact of each feature.
- **特征排序**：根据系数大小对特征进行排序，找出对评分影响最大的因素。
  - **Feature Ranking**: Sort features based on the size of their coefficients to identify the factors that have the greatest impact on ratings.

### 步骤 6: 结果应用
### Step 6: Results Application
- **结果解读**：根据模型结果解释哪些方面对产品评分有显著影响。
  - **Results Interpretation**: Interpret the model results to explain which aspects significantly impact product ratings.


## 注意事项
## Considerations
- **数据多样性**：确保数据集在类型、用户群体等方面具有代表性。
  - **Data Diversity**: Ensure that the dataset is representative in terms of type, user demographics, etc.
- **模型假设**：线性回归模型假设特征与响应变量之间存在线性关系，这可能不完全适用于所有情况。
  - **Model Assumptions**: Linear regression models assume a linear relationship between features and the response variable, which may not hold in all cases.
- **解释性**：BoW 模型简单但忽略了词序和上下文信息，可能影响对评论内容的完全理解。
  - **Interpretability**: Although the BoW model is simple, it ignores word order and context, which may affect the full understanding of review content.
