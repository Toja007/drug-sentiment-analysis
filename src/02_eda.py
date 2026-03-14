#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re


# In[2]:


#loading the datasets
training_path = r'C:\\Users\\GIGABYTE\\Desktop\\drug-sentiment-analysis\\data\\raw\\train_F3WbcTw_1icmK82.csv'
test_path = r'C:\\Users\\GIGABYTE\\Desktop\\drug-sentiment-analysis\\data\\raw\\test_tOlRoBf_VeRQtHl.csv'
training_data = pd.read_csv(training_path)
test_data = pd.read_csv(test_path)


# In[17]:


#display sentiment distribution per drug
training_data.groupby(['drug','sentiment']).size().unstack().fillna(0).plot(
    kind='bar',
    stacked=True,
    figsize=(20,6)
)
plt.title("Sentiment Distribution per Drug")
plt.show()


# In[9]:


training_data[training_data["drug"] == 'almita']['text'].iloc[0]


# In[18]:


training_data[training_data["drug"] == 'alimta']['text'].iloc[0]


# In[19]:


#distribution of comment length
training_data['comment_length'] = training_data['text'].apply(len)
sns.histplot(training_data['comment_length'], bins=50)
plt.title("Distribution of Comment Length")
plt.show()


# In[ ]:


#comment length by sentiment
sns.boxplot(x='sentiment', y='comment_length', data=training_data)
plt.title("Comment Length by Sentiment")
plt.show()


# In[ ]:


#finding the most common words
from collections import Counter

words = " ".join(training_data['text']).split()
Counter(words).most_common(20)


# In[ ]:


#ploting word cloud for the most common words
from wordcloud import WordCloud

text = " ".join(training_data['text'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Most Frequent Words in Reviews")
plt.show()


# In[ ]:


#ploting word cloud for the most common words in negative sentiment
negative_text = " ".join(training_data[training_data['sentiment']==1]['text'])

wordcloud = WordCloud(width=800, height=400).generate(negative_text)

plt.imshow(wordcloud)
plt.axis('off')
plt.title("Common Words in Negative Reviews")
plt.show()


# In[ ]:


#ploting word cloud for the most common words in positive sentiment
positive_text = " ".join(training_data[training_data['sentiment']==0]['text'])

wordcloud = WordCloud(width=800, height=400).generate(positive_text)

plt.imshow(wordcloud)
plt.axis('off')
plt.title("Common Words in Positive Reviews")
plt.show()


# In[ ]:


#ploting word cloud for the most common words in neutral sentiment
neutral_text = " ".join(training_data[training_data['sentiment']==2]['text'])

wordcloud = WordCloud(width=800, height=400).generate(neutral_text)

plt.imshow(wordcloud)
plt.axis('off')
plt.title("Common Words in Neutral Reviews")
plt.show()


# # Observation
# 1. Presence of Special Characters in the Text
# 
# The comments in the text column contain various special characters such as punctuation marks, symbols, and irregular formatting. These characters do not contribute meaningful information for sentiment classification and may introduce noise into the model.
# 
# Implication:
# These characters will be removed during the text preprocessing stage to improve the quality of the textual data.
# 
# 2. Inconsistent Drug Name Spellings
# 
# Some drug names appear to be misspelled both in the drug column and within the text column. These inconsistencies may cause the same drug to be treated as multiple different entities.
# 
# Implication:
# Drug names will be standardized during preprocessing to ensure that all references to the same drug are consistent across the dataset.
# 
# 3. High Frequency of Stopwords
# 
# The most frequently occurring words in the dataset include common stopwords such as "the", "and", "to", and "is". These words occur frequently but typically carry little semantic meaning for sentiment analysis.
# 
# Implication:
# Stopwords will be removed during preprocessing to reduce noise and improve model performance.
# 
# Optional 4th Observation (Very Good to Add)
# 
# If you noticed comment length variation:
# 
# 4. Variation in Comment Length
# 
# The length of comments varies significantly across the dataset, with some reviews being very short while others are detailed descriptions of user experiences.
# 
# Implication:
# This variation may influence the richness of information available for sentiment classification.
