# Enhancing Product Review Insights with OpenAI API

## Overview
Leveraging OpenAI's advanced NLP capabilities, this project aims to automate the extraction and analysis of product review content, focusing initially on the camera category. Our approach encompasses two primary pipelines: a summarization pipeline and an aspect-based evaluation pipeline, designed to provide concise summaries and detailed aspect analysis of product reviews, respectively. This initiative seeks to transform unstructured review data into structured insights, aiding both consumers and businesses in making informed decisions.

## Team Members
- Calla Gong: BS in AMS and BA in CS, calla.gong@emory.edu
- Louis Lu: BS in AMS and BS in CS, louis.lu@emory.edu
- Wenzhuo Ma: BS in CS, wma44@emory.edu
- Yoyo Wang: BS in AMS and BA in CS, hwan592@emory.edu

## Objectives
- To conduct a comprehensive analysis of the Amazon Product Review dataset, identifying key attributes contributing to review helpfulness.
- To develop data pipelines using the OpenAI API for summarizing customer reviews and performing aspect-based evaluation.

## Motivation
Enhancing the shopping experience and promoting transparency in the online marketplace by analyzing Amazon product reviews to empower consumers with more informed purchasing decisions.

## Installation

Follow these steps to set up the project environment:

```bash
# Clone the repository
git clone https://github.com/mawenzhuo2022/Analyze_Amazon_Review.git

# Navigate to the project directory
cd Analyze_Amazon_Review

# Install required Python packages
pip install -r requirements.txt
```

## Methodology

### Data Collection and Preprocessing
- **Data Acquisition**: Compilation of the Amazon product review dataset.
- **Data Cleaning**: Preprocessing steps to clean and prepare the dataset for analysis.

### Summarization Pipeline
- Utilizes the OpenAI API for NLP processing to generate concise summaries of product reviews.

### Aspect-Based Evaluation Pipeline
- Employs OpenAI API for detailed analysis, identifying and scoring specific product attributes mentioned in reviews.

## Datasets
The project uses the Amazon product review dataset, encompassing a wide range of consumer feedback across various product categories. This dataset is instrumental in our analysis, providing a robust foundation for developing and refining our pipelines.

## Evaluation Methods
- **Accuracy of Summarization**: The accuracy of the generated summaries is compared against human-crafted abstracts to ensure that the key points and sentiments of the original reviews are accurately captured.
- **Precision of Aspect Identification**: The effectiveness of the aspect-based evaluation pipeline is measured by its ability to correctly identify and score the product aspects mentioned in reviews, using a human-annotated dataset as a benchmark.

## Timeline
- **Week 1**: Data preprocessing & API setup.
- **Weeks 2-3**: Designing the prompt for the first pipeline.
- **Weeks 3-4**: Designing the prompt for the second pipeline.
- **Week 5**: Testing on different products.

## Acknowledgements
Special thanks to the contributors of large language models, including ChatGPT, for enhancing the readability and structure of this project proposal.

We also extend our gratitude to Dr. Jinho Choi, associate professor of Computer Science, Quantitative Theory and Methods, and Linguistics at Emory University, for his invaluable insights and contributions to the field of natural language processing, which have inspired aspects of this project.

## References
- AlQahtani, Arwa S. M. (2021). "Product sentiment analysis for amazon reviews." SSRN. [Access here](http://papers.ssrn.com/sol3/papers.cfm?abstract_id=3886135).
- Tan, Wanliang, et al. (2018). "Sentiment analysis for amazon review." Stanford University. [Access here](http://cs229.stanford.edu/proj2018/report/122.pdf).

