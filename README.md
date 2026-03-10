# Drug Sentiment Analysis

## Overview

Sentiment analysis is a Natural Language Processing (NLP) technique used to determine the emotional tone behind textual data. It helps organizations automatically analyze large volumes of customer opinions, feedback, and reviews.

In the pharmaceutical industry, sentiment analysis can provide valuable insights into how patients perceive medications, including their effectiveness and potential side effects.

This project focuses on building a **drug-specific sentiment analysis model** capable of identifying whether a comment about a particular drug expresses **positive, negative, or neutral sentiment**.

The model is designed to analyze textual comments scraped from various online platforms and determine the sentiment associated with specific drugs mentioned within those comments.

---

## Problem Statement

Pharmaceutical company **Kipla** has collected large amounts of user comments about different medications from various online sources. These comments often contain opinions about multiple drugs within the same text.

The company wants to build a **sentiment analysis engine** that can automatically track the sentiment associated with each drug.

Each comment may mention one or more drugs, and the sentiment toward each drug may differ.

For example:

> "I looked up stomach pain after taking Correctol and experienced severe abdominal pain. I will never take this medication again. Later I took Meftal Spas and it worked wonders for me."

In this example:

* Sentiment toward **Correctol** is **Negative**
* Sentiment toward **Meftal Spas** is **Positive**

The objective is to build a machine learning model that can predict the sentiment (**Positive, Negative, Neutral**) of a given drug mentioned in a comment.

---

## Objectives

The main objectives of this project are:

* Perform exploratory data analysis (EDA) on drug review data
* Clean and preprocess textual data
* Transform text into numerical representations using NLP techniques
* Train and evaluate machine learning models for sentiment classification
* Build a model capable of predicting drug-specific sentiment from comments

---

## Dataset Source

The dataset used in this project was obtained from the **Analytics Vidhya Data Science platform**.

Source:
Analytics Vidhya – Drug Sentiment Analysis Challenge

The dataset contains:

* Drug names
* User comments
* Sentiment labels (0-positive, 1-negative, 2-neutral)

---

## Project Workflow

The project follows a standard Natural Language Processing pipeline:

1. Data Understanding and Exploration
2. Text Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Sentiment Prediction

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Machine Learning Models

---

## Expected Outcome

A trained machine learning model capable of predicting the sentiment associated with a specific drug mentioned in a comment.

---

## Author

Tochukwu Umunnakwe

