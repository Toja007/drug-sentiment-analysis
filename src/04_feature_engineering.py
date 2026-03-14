#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import neessary libraries for feature engineering
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix,hstack
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder


# In[2]:


#load the processed datasets
training = pd.read_csv(r'C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\training_data.csv')
test = pd.read_csv(r'C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\test_data.csv')


# In[3]:


training.head()


# In[4]:


# check number of unique words in the cleaned_text column
unique_words = set(" ".join(training['cleaned_text']).split())

print("Vocabulary size:", len(unique_words))


# In[5]:


from collections import Counter

all_words = " ".join(training['cleaned_text']).split()

word_counts = Counter(all_words)

rare_words = [word for word, count in word_counts.items() if count == 1]

print("Rare words:", len(rare_words))


# # Observation
# 
# The dataset contains 40,035 unique words, out of which 17,396 words appear only once across all reviews. This indicates a high proportion of rare tokens, which is common in natural language datasets due to spelling variations, domain-specific terminology, and noise.
# 
# To reduce dimensionality and improve model generalization, rare words will be filtered during TF-IDF feature extraction using a minimum document frequency threshold (min_df).
# 
# Some drugs appeared very infrequently in the dataset, with certain drugs occurring only once. Such rare categories can introduce noise when using one-hot encoding because the model cannot learn reliable patterns from a single observation. To mitigate this, drugs appearing fewer than a specified threshold were grouped into an other_drug category before encoding.

# In[6]:


# vectorizer
tfidf = TfidfVectorizer(
    max_features=6000,
    ngram_range=(1,2),
    min_df=5,
    max_df=0.9
)


# In[7]:


# checking rare drugs 
drug_counts = training['clean_drug'].value_counts()

rare_drugs = drug_counts[drug_counts < 5].index


# In[8]:


rare_drugs


# In[9]:


# repalcing rare drugs with "other drugs"
training['clean_drug'] = training['clean_drug'].replace(rare_drugs, 'other_drug')
test['clean_drug'] = test['clean_drug'].replace(rare_drugs, 'other_drug')


# In[10]:


#get only the needed columns for feature engineering
train = training[["cleaned_text", "clean_drug", "sentiment"]]
testing = test[["cleaned_text", "clean_drug"]]


# In[11]:


# Drug mention feature
train['drug_in_text'] = train.apply(
    lambda x: 1 if x['clean_drug'] in x['cleaned_text'] else 0,
    axis=1

)

testing['drug_in_text'] = testing.apply(
    lambda x: 1 if x['clean_drug'] in x['cleaned_text'] else 0,
    axis=1
)


# In[ ]:


# defining list of positive and negative words for feature enginnering
positive_words = [
    "worked","effective","great","excellent","relief",
    "helped","improved","better","amazing","perfect", "improvement"
    "recommended","success","good","love","progress","breakthrough",
    "promising", "best", "hope", "positive", "fantastic", "benefit"
]

negative_words = [
    "pain","nausea","vomit","vomiting","dizziness",
    "fatigue","rash","headache","diarrhea","cramps",
    "terrible","awful","worse","bad","horrible", "failure",
    "miserable", "fail", "horrifying", "infection", "terrified" 'terrible',
    "coughing", "insane"
]


# In[13]:


train[train['sentiment'] == 1]['cleaned_text'].iloc[10]


# In[ ]:


# function to count positive words
def positive_score(text):
    words = text.split()
    return sum(word in positive_words for word in words)

train['positive_score'] = train['cleaned_text'].apply(positive_score)
testing['positive_score'] = testing['cleaned_text'].apply(positive_score)


# In[ ]:


#function to count negative words
def negative_score(text):
    words = text.split()
    return sum(word in negative_words for word in words)

train['negative_score'] = train['cleaned_text'].apply(negative_score)
testing['negative_score'] = testing['cleaned_text'].apply(negative_score)


# In[ ]:


# create sentiment strength features
train['sentiment_strength'] = train['positive_score'] - train['negative_score']
testing['sentiment_strength'] = testing['positive_score'] - testing['negative_score']


# In[ ]:


#list of side effects
side_effect_words = [
    "side effect","caused","reaction","bleeding",
    "nausea","dizziness","headache", "fatigue", "diarrhea", "abdominal pain", "vomiting", "constipation", "dry mouth", "insomnia", 
    "rash", "itching", "swelling", "difficulty breathing", "chest pain", "irregular heartbeat", "seizures", 
    "muscle pain", "joint pain", "blurred vision", "hearing loss", "anxiety", "depression"
]


# In[ ]:


# function to check for side effects
def side_effect_flag(text):
    return int(any(word in text for word in side_effect_words))

train['side_effect_flag'] = train['cleaned_text'].apply(side_effect_flag)
testing['side_effect_flag'] = testing['cleaned_text'].apply(side_effect_flag)


# In[ ]:


#list of words indicating drug effectiveness
effect_words = [
    "worked","relief","effective","improved","helped"
]


# In[ ]:


# fuc t0 check for drug effectiveness
def effectiveness_flag(text):
    return int(any(word in text for word in effect_words))

train['effect_flag'] = train['cleaned_text'].apply(effectiveness_flag)
testing['effect_flag'] = testing['cleaned_text'].apply(effectiveness_flag)


# In[ ]:


#list of neutral words
neutral_words = [
    "started","taking","prescribed","week","month","day"
]

def neutral_score(text):
    words = text.split()
    return sum(word in neutral_words for word in words)

train['neutral_score'] = train['cleaned_text'].apply(neutral_score)
testing['neutral_score'] = testing['cleaned_text'].apply(neutral_score)


# In[ ]:


#create pos to neg ratio feature
train['pos_neg_ratio'] = train['positive_score'] / (train['negative_score'] + 1)
testing['pos_neg_ratio'] = testing['positive_score'] / (testing['negative_score'] + 1)


# In[ ]:


#checking the engineered features
train.head()


# In[24]:


train.columns


# In[ ]:


#one hot encoding foor drugs
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)


# define feature and target variables
X_text = train['cleaned_text']
X_drug = one_hot_encoder.fit_transform(train[["clean_drug"]])
y = train['sentiment']


# In[33]:


#vectorize the text data
X_tfidf = tfidf.fit_transform(X_text)


# In[36]:


# convert one-hot encoding to sparese marix

X_drug_sparse = csr_matrix(X_drug)


# In[ ]:


# combine the text and drug features

X = hstack([
    X_tfidf,
    X_drug_sparse,
    csr_matrix(train[['drug_in_text',
                      'positive_score',
                      'negative_score',
                      'sentiment_strength',
                      'side_effect_flag',
                      'effect_flag',
                      'neutral_score',
                      'pos_neg_ratio']].values)
])


# In[38]:


X.shape


# In[39]:


y.shape


# In[ ]:


# over sampling the minority class
ros = RandomOverSampler(
    sampling_strategy={0:1200, 1:1200}
)

X_over, y_over = ros.fit_resample(X, y)


# In[ ]:


# undersampling the majority class
rus = RandomUnderSampler(
    sampling_strategy={2:2000}
)

X_resampled, y_resampled = rus.fit_resample(X_over, y_over)


# In[42]:


X_resampled.shape


# In[43]:


X.shape


# In[44]:


#spliting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)


# In[45]:


#feature engineering for the test dataset
testing_text = testing['cleaned_text']
testing_drug = one_hot_encoder.transform(testing[["clean_drug"]])


testing_tfidf = tfidf.transform(testing_text)
#testing_drug_encoded = one_hot_encoder.transform(testing_drug)
testing_drug_sparse = csr_matrix(testing_drug)
final_testing = hstack([testing_tfidf, testing_drug_sparse,csr_matrix(testing[['drug_in_text',
                      'positive_score',
                      'negative_score',
                      'sentiment_strength',
                      'side_effect_flag',
                      'effect_flag',
                      'neutral_score',
                      'pos_neg_ratio']].values)])


# In[46]:


final_testing.shape


# In[47]:


X.shape


# In[49]:


import joblib

# Save training features and target
joblib.dump(X_train, r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed/X_train.pkl")
joblib.dump(X_val, r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed/X_val.pkl")
joblib.dump(y_train, r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed/y_train.pkl")
joblib.dump(y_val, r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed/y_val.pkl")

# Save test features
joblib.dump(final_testing, r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed/X_test.pkl")
joblib.dump(tfidf, r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed/tfidf_vectorizer.pkl")
#joblib.dump(X_drug_encoded.columns, r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed/drug_columns.pkl")

