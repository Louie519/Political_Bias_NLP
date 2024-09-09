# Political Bias Detection using NLP

## Overview

This project explores detecting political bias using Natural Language Processing (NLP) techniques. We utilized data from Reddit comments on political subreddits and news articles from politically biased platforms. The goal was to classify political bias (left, center, right) in news articles and apply these models to classify Reddit comments by political bias. The project includes the use of multiple models such as Logistic Regression, SVM, and BERT for bias classification, and GPT-based models for text generation.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Collection](#data-collection)
   - [Reddit Scraping](#reddit-scraping)
   - [News Articles Collection](#news-articles-collection)
3. [Modeling Approach](#modeling-approach)
   - [Logistic Regression](#logistic-regression)
   - [SVM Classifier](#svm-classifier)
   - [AdaBoost Classifier](#adaboost-classifier)
   - [BERT-based Model](#bert-based-model)
4. [Evaluation and Classification Scores](#evaluation-and-classification-scores)
5. [Text Generation with GPT](#text-generation-with-gpt)
   - [Reddit Subreddit Comments](#reddit-subreddit-comments)
   - [News Articles](#news-articles)
6. [Challenges and Findings](#challenges-and-findings)
7. [Future Considerations](#future-considerations)

## Introduction

The goal of this project was to detect political bias in online content, specifically Reddit comments and news articles, using NLP techniques. We scraped data from political discourse subreddits and news platforms, trained classification models based on bias labels from AllSides, and attempted to apply these models to classify Reddit comments by political bias.

## Data Collection

### Reddit Scraping

We collected posts and comments from a variety of political subreddits such as `socialism`, `conservatives`, and `progressive`. We used [Reddits developer API](https://www.reddit.com/dev/api/) for data collection. The scraped data included:
- Post titles, content, scores, subreddit names
- Associated comments

### News Articles Collection

We used the [**Fundus**](https://github.com/flairNLP/fundus) library to crawl and collect articles from politically biased news platforms, including `The Gateway Pundit`, `Reuters`, and `Wired`. The collected dataset contained:
- Article titles and plaintext content
- Publisher names and AllSides bias ratings (left, center, right)

## Modeling Approach

### Logistic Regression

We started by using a **Logistic Regression model** as a baseline classifier to detect political bias. For feature extraction, we used **Count Vectorizer** and **TF-IDF Vectorizer** to represent the text data.

- **Vectorizers**: 
  - **Count Vectorizer**: Converts a collection of text into a matrix of token counts.
  - **TF-IDF Vectorizer**: Transforms the text into a matrix of term frequency and inverse document frequency, emphasizing less frequent but informative words.

- **Training**: We trained the logistic regression model on the news articles labeled with political bias (left, center, right).
  
- **Results**: The Logistic Regression model performed reasonably well in detecting political bias from news articles. However, it struggled with classifying Reddit comments due to the informal and less structured nature of the text.

### SVM Classifier

Next, we trained a **Support Vector Machine (SVM)** classifier, which is well-suited for high-dimensional data like text. We experimented with both **Count Vectorizer** and **TF-IDF Vectorizer** for feature extraction, followed by hyperparameter tuning using grid search.

- **SVM Parameters**: We performed grid search for the regularization parameter `C` and the kernel coefficient `gamma`.
  
- **Training**: The SVM model was trained on the same news article dataset and validated using a test set.
  
- **Results**: The SVM classifier with the **TF-IDF Vectorizer** performed better than the one with the **Count Vectorizer**.

### AdaBoost Classifier

We also experimented with **AdaBoost** as a classifier. This ensemble method combines multiple weak learners to improve the overall accuracy.

- **Training**: The AdaBoost classifier was trained on the same dataset, using both **Count Vectorizer** and **TF-IDF Vectorizer** representations.

### BERT-based Model

We also implemented a **BERT-based model (DistilBERT)** for political bias classification. BERT models capture the contextual meaning of words and are highly effective in NLP tasks. We fine-tuned the **DistilBERT** model on the news articles to classify them as left, center, or right.

- **Training**: The fine-tuned DistilBERT model was trained for 6 epochs on the same news article dataset.
  
- **Results**: The BERT-based model achieved the highest accuracy and was able to capture nuanced relationships in the text.

## Evaluation and Classification Scores

Here are the evaluation metrics (precision, recall, f1-score) for the models we used:

### SVM with TF-IDF Results:
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Center    | 0.85      | 0.87   | 0.86     | 90      |
| Left      | 0.81      | 0.91   | 0.86     | 90      |
| Right     | 0.87      | 0.74   | 0.80     | 90      |
| **Accuracy**      | **84%**        |            |          | 270     |
| **Macro Avg**     | 0.84      | 0.84   | 0.84     | 270     |
| **Weighted Avg**  | 0.84      | 0.84   | 0.84     | 270     |

### SVM with Count Vectorizer Results:
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Center    | 0.88      | 0.83   | 0.86     | 90      |
| Left      | 0.75      | 0.77   | 0.76     | 90      |
| Right     | 0.77      | 0.80   | 0.79     | 90      |
| **Accuracy**      | **80%**        |            |          | 270     |
| **Macro Avg**     | 0.80      | 0.80   | 0.80     | 270     |
| **Weighted Avg**  | 0.80      | 0.80   | 0.80     | 270     |

### AdaBoost with Count Vectorizer Results:
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Class 0   | 0.66      | 0.88   | 0.75     | 90      |
| Class 1   | 0.87      | 0.71   | 0.78     | 86      |
| Class 2   | 0.84      | 0.71   | 0.77     | 94      |
| **Accuracy**      | **77%**        |            |          | 270     |
| **Macro Avg**     | 0.79      | 0.77   | 0.77     | 270     |
| **Weighted Avg**  | 0.79      | 0.77   | 0.77     | 270     |

### AdaBoost with TF-IDF Results:
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Class 0   | 0.69      | 0.87   | 0.77     | 90      |
| Class 1   | 0.87      | 0.78   | 0.82     | 86      |
| Class 2   | 0.89      | 0.76   | 0.82     | 94      |
| **Accuracy**      | **80%**        |            |          | 270     |
| **Macro Avg**     | 0.82      | 0.80   | 0.80     | 270     |
| **Weighted Avg**  | 0.82      | 0.80   | 0.80     | 270     |

### BERT Results:
| Metric         | BERT-based Model |
|----------------|------------------|
| **Accuracy**   | **85.0%**        |
| **Test Loss**  | 0.4339           |

## Text Generation with GPT

### News Articles

We fine-tuned **GPT-based models** on news articles from each political bias category (left, center, right). The goal was to generate contextually accurate text that reflects the political bias of the training data.

### Reddit Subreddit Comments

For Reddit comments, we used a similar fine-tuning approach with GPT. However, due to the short and informal nature of comments, the GPT model generated more direct, opinion-driven outputs.

## Challenges and Findings

### Challenges

- **Contextual Nature of Reddit Comments**: Reddit comments were challenging to classify due to their conversational style and lack of clear political alignment. Additional context or metadata might be needed to improve classification.
  
- **Limited Labeled Data**: The limited amount of manually labeled Reddit data made it difficult to train effective classifiers.

### Findings

- **Text Generation**: GPT models trained on news articles were able to generate more structured, nuanced responses compared to those trained on Reddit comments.
  
- **Political Bias Detection**: The BERT-based model was the most effective at detecting political bias in structured news articles, but further work is needed to apply these models to Reddit comments.

## Future Considerations

1. **Improve Reddit Comment Classification**: Consider classifying entire Reddit threads rather than individual comments, which might provide more context and improve classification accuracy.
  
2. **Increase Dataset Size**: Expand the labeled dataset, particularly for Reddit comments, to improve model performance.

3. **Fine-Tune GPT for Reddit**: Further fine-tuning GPT models on larger Reddit datasets may improve the text generation for online discussions.

