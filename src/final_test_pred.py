#!/usr/bin/env python
# coding: utf-8

# # final prection
# 1. load text csv
# 2. load prepeocess text data
# 3. load model
# 4. make prediction
# 5. convert prediction to csv with header sentiment_prediction
# 6. merge csv prediction with text csv
# 7. save predcition

# In[1]:


# import libraries
import joblib
import numpy as np
import pandas as pd


# In[2]:


#load text cvs, processed test and all model
test_df = pd.read_csv(r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\test_data.csv")
#X_test = joblib.load(r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\X_test.pkl")
X_test = joblib.load(r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\X_test.pkl")


# In[4]:


from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# In[5]:


class BERTClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, dropout=0.5):
        super(BERTClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# In[6]:


device = torch.device("cuda")


# In[8]:


model = BERTClassifier(input_dim=X_test.shape[1])
model.load_state_dict(torch.load(r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\model\nn_model.pt"))
model.to(device)
model.eval()


# In[9]:


X_test_tensor = torch.tensor(X_test, dtype=torch.float).to(device)


# In[10]:


with torch.no_grad():
    outputs = model(X_test_tensor)
    preds = torch.argmax(outputs, dim=1)


# In[11]:


preds = preds.cpu().numpy()


# In[12]:


import pandas as pd

pred_df = pd.DataFrame(preds, columns=["prediction"])


# In[13]:


nn_pred = pd.DataFrame({
    "id" : test_df["unique_hash"],
    "sentiment" : preds
    })


# In[14]:


nn_pred.to_csv(r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\predictions\nn_pred.csv", index=False)


# In[22]:


knn = joblib.load(r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\model\knn.pkl")


# In[23]:


#making prediction
def make_pred(model,X_test):
    predictions = model.predict(X_test)
    return predictions


# In[24]:


# convert predicgtions csv
def convert_to_csv(predictions, test_df):
    submission_df = pd.DataFrame({
    "id" : test_df["unique_hash"],
    "sentiment" : predictions
    })
    return submission_df


# In[25]:


# make prediction for decision tree
knn_pred = make_pred(knn,X_test)

knn_df = convert_to_csv(knn_pred, test_df)

knn_df.to_csv(r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\predictions\knn_pred.csv", index=False)


# In[ ]:




