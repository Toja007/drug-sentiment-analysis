#!/usr/bin/env python
# coding: utf-8

# In[4]:


#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re


# In[8]:


#loading the datasets
training_path = r'C:\\Users\\GIGABYTE\\Desktop\\drug-sentiment-analysis\\data\\raw\\train_F3WbcTw_1icmK82.csv'
test_path = r'C:\\Users\\GIGABYTE\\Desktop\\drug-sentiment-analysis\\data\\raw\\test_tOlRoBf_VeRQtHl.csv'
training_data = pd.read_csv(training_path)
test_data = pd.read_csv(test_path)


# In[9]:


#see first 5 rows of dataset
training_data.head()


# In[ ]:


#checking the shape of the datasets
print('training data shape:', training_data.shape)
print('test data shape:', test_data.shape)


# In[15]:


#info about the dataset
training_data.info()


# In[17]:


#checking for missing values
training_data.isnull().sum()


# In[12]:


#value counts of target variable
training_data['sentiment'].value_counts()


# we have imbalanced target data

# In[14]:


#ploting the distribution of target variable
sns.countplot(x='sentiment', data=training_data)
plt.title('sentiment distribution');


# In[19]:


#checking drugs mentioned
training_data["drug"].value_counts().head(10)


# In[23]:


training_data["drug"].value_counts().tail(10)


# In[24]:


training_data["drug"].nunique()


# In[22]:


training_data.sample(5)['text'].values


# In[25]:


training_data['text'][0]


# In[ ]:


#checking for duplicates
training_data.duplicated().sum()


# In[45]:


training_data.groupby('drug')['sentiment'].value_counts().unstack().fillna(0).div(training_data.groupby('drug')['sentiment'].value_counts().unstack().fillna(0).sum(axis=1), axis=0).head(20)


# In[44]:


top_drugs = training_data['drug'].value_counts().head(20).index

training_data[training_data['drug'].isin(top_drugs)] \
.groupby(['drug','sentiment']) \
.size() \
.unstack() \
.fillna(0) \
.plot(kind='bar', stacked=True, figsize=(24,6))


# In[47]:


training_data['drug'].value_counts().head(20).plot(kind='bar')


# In[48]:


from collections import Counter

all_words = " ".join(training_data['text']).split()
Counter(all_words).most_common(20)


# In[50]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = " ".join(training_data['text'])

wordcloud = WordCloud(width=800, height=400).generate(text)

plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# # observation
# 
# 1. Dataset Size
# 
# The training dataset contains 5,279 observations, each representing a user comment associated with a specific drug and its corresponding sentiment label.
# 
# Implication:
# This dataset size is relatively small for deep learning models, which suggests that traditional machine learning algorithms may perform well.
# 
# 2. Number of Unique Drugs
# 
# There are 102 unique drugs present in the training dataset.
# 
# Implication:
# This indicates that the dataset covers a wide range of medications, and sentiment patterns may vary across different drugs.
# 
# 3. Imbalanced Sentiment Distribution
# 
# The sentiment column shows a class imbalance, where one sentiment class appears significantly more frequently than the others.
# 
# Implication:
# use stritify when splitting data
# Class imbalance may affect model performance, so evaluation metrics such as F1-score will be important when comparing models.
# 
# 
# 4. Neutral Sentiment Dominates
# 
# A large proportion of the comments express neutral sentiment toward the drugs.
# 
# Implication:
# The model may become biased toward predicting the neutral class if class imbalance is not considered during training.
# 
# 5. Drug Frequency Distribution
# 
# The drug "Ocrevus" appears most frequently in the dataset, making it the most discussed medication in the training data.
# 
# Implication:
# Certain drugs may dominate the dataset, which could influence model learning if not properly considered.
