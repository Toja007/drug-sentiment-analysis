#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary libraries for text preprocessing
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# In[2]:


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


# In[3]:


#loading the datasets
training_path = r'C:\\Users\\GIGABYTE\\Desktop\\drug-sentiment-analysis\\data\\raw\\train_F3WbcTw_1icmK82.csv'
test_path = r'C:\\Users\\GIGABYTE\\Desktop\\drug-sentiment-analysis\\data\\raw\\test_tOlRoBf_VeRQtHl.csv'
training_data = pd.read_csv(training_path)
test_data = pd.read_csv(test_path)


# In[4]:


#mapping the wrongly splelled drug names
drug_corrections = {
    "panrentinal photocoagulation": "pan-retinal photocoagulation",
    "pan-rentinal photocoagulation": "pan-retinal photocoagulation",

    "nivolumabb": "nivolumab",

    "ketruda": "keytruda",

    "giotrif": "gilotrif",

    "crizotnib": "crizotinib",

    "almita": "alimta"
}


# In[5]:


#text preprocessing fuction
def preprocess_text(text):
    #lowercase
    text = text.lower()

    #remove urls
    text = re.sub(r'http\S+/www\S+', '', text)

    #remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    #tokenization
    tokens = word_tokenize(text)

    #stop word removal
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]

    #lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    #join tokens back to string
    text = " ".join(tokens)
    return text


# In[6]:


#function to correct drug names in the text
def correct_drugs(text):
    for wrong, correct in drug_corrections.items():
        text = text.replace(wrong, correct)
    return text


# In[7]:


#function to correct drug names in drug column
def correct_drug_col(drug):
    for wrong, correct in drug_corrections.items():
        drug = drug.replace(wrong, correct)
    return drug


# In[8]:


#apply preprocessing on training and test datasets
training_data["cleaned_text"] = training_data["text"].apply(preprocess_text)

test_data["cleaned_text"] = test_data["text"].apply(preprocess_text)

#correct drug names in drug column
training_data['clean_drug'] = training_data['drug'].apply(correct_drug_col)

test_data['clean_drug'] = test_data['drug'].apply(correct_drug_col)


# In[9]:


training_data["cleaned_text"] = training_data["cleaned_text"].apply(correct_drugs)
test_data["cleaned_text"] = test_data["cleaned_text"].apply(correct_drugs)


# In[10]:


training_data.text[0], training_data.cleaned_text[0]


# In[15]:


training_data[training_data["drug"] == 'ketruda']['text']


# In[16]:


training_data[training_data["clean_drug"] == 'ketruda']['text']


# In[17]:


#save the preprocessed datasets
training_data.to_csv(r'C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\training_data.csv', index=False)

test_data.to_csv(r'C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\test_data.csv', index=False)


# In[ ]:




