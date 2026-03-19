#!/usr/bin/env python
# coding: utf-8

# In[23]:


#import neessary libraries for feature engineering
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix,hstack
from imblearn.over_sampling import RandomOverSampler
from transformers import BertTokenizer, BertModel
import torch
import joblib


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

# In[7]:


# initialize the Bert tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

device = torch.device("cuda")
model.to(device)
model.eval()


# In[8]:


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)


# In[9]:


def get_bert_embeddings(texts, batch_size=32):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        encoded_inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )


        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}

        with torch.no_grad():
            outputs = model(**encoded_inputs)


        batch_embeddings = mean_pooling(
            outputs.last_hidden_state,
            encoded_inputs['attention_mask']
        )

        all_embeddings.append(batch_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


# In[10]:


training['final_text'] = training["cleaned_text"] + " " + training["clean_drug"]
test["final_text"] = test["cleaned_text"] + " " + test["clean_drug"]


# In[11]:


X_train_bert = get_bert_embeddings(training["final_text"].tolist())
X_test_bert = get_bert_embeddings(test["final_text"].tolist())


# In[12]:


y = training["sentiment"]


# In[17]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_bert = scaler.fit_transform(X_train_bert)
X_test_bert = scaler.transform(X_test_bert)


# In[29]:


ros = RandomOverSampler()
X_train_bert, y = ros.fit_resample(X_train_bert, y)



# In[30]:


X_train, X_val, y_train, y_val = train_test_split(
    X_train_bert,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[32]:


X_train.shape, X_val.shape


# In[33]:


y_train.shape,y_val.shape


# In[34]:


joblib.dump(X_train, r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\X_train.pkl")
joblib.dump(X_val, r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\X_val.pkl")
joblib.dump(y_train, r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\y_train.pkl")
joblib.dump(y_val, r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\y_val.pkl")

joblib.dump(X_test_bert, r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\X_test.pkl")


# In[ ]:




