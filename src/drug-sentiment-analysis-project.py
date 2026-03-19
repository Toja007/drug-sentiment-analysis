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





# In[1]:


# import libraries for model training and evaluation
import joblib
import numpy as mp
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


# In[2]:


#read in the processed data
X_train = joblib.load(r'C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\X_train.pkl')
y_train = joblib.load(r'C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\y_train.pkl')

X_val = joblib.load(r'C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\X_val.pkl')
y_val = joblib.load(r'C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\y_val.pkl')


# In[3]:


# train and evaluate logistic regression model as baseline
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train, y_train)


# In[4]:


#make predictiona on test set
predictions = lr.predict(X_val)

#print f1 score on test set
print(f"the f1 score for logistic regression on the test set is: {f1_score(y_val, predictions, average='macro')}")
print(f"The f1 score for logistic regression on the training set is: {f1_score(y_train, lr.predict(X_train), average='macro')}")


# In[5]:


print(classification_report(y_val, predictions))


# In[6]:


dt_model = DecisionTreeClassifier(
    class_weight='balanced',
    max_depth=2,          
    min_samples_leaf=5     
)

dt_model.fit(X_train, y_train)


# In[7]:


#make prediction on test set with decision tree
predictions_dt = dt_model.predict(X_val)

#print f1 score on test set
print(f"the f1 score for decision tree on the test set is: {f1_score(y_val, predictions_dt, average='macro')}")
print(f"The f1 score for decision tree on the training set is: {f1_score(y_train, dt_model.predict(X_train), average='macro')}")


# In[8]:


print(classification_report(y_val, predictions_dt))


# In[9]:


rf_model = RandomForestClassifier(
    n_estimators=5,
    class_weight='balanced',
    max_depth=3,
    random_state=42
)

rf_model.fit(X_train, y_train)


# In[10]:


#make prediction on test set with random forest
predictions_rf = rf_model.predict(X_val)

#print f1 score on test set
print(f"the f1 score for random forest on the test set is: {f1_score(y_val, predictions_rf, average='macro')}")
print(f"The f1 score for random forest on the training set is: {f1_score(y_train, rf_model.predict(X_train), average='macro')}")


# In[11]:


print(classification_report(y_val, predictions_rf))


# In[12]:


xgb_model = XGBClassifier(
    objective='multi:softprob',   
    num_class=3,                  
    eval_metric='mlogloss',
    n_estimators=50,
    learning_rate=0.1,
    max_depth=2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)


# In[13]:


#make prediction on test set with random forest
predictions_xgb = xgb_model.predict(X_val)

#print f1 score on test set
print(f"the f1 score for xgb on the test set is: {f1_score(y_val, predictions_xgb, average='macro')}")
print(f"The f1 score for xgb on the training set is: {f1_score(y_train, xgb_model.predict(X_train), average='macro')}")


# In[14]:


print(classification_report(y_val, predictions_xgb))


# In[15]:


from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)


# In[16]:


#make prediction on test set with random forest
predictions_knn = knn_model.predict(X_val)

#print f1 score on test set
print(f"the f1 score for knn on the test set is: {f1_score(y_val, predictions_knn, average='macro')}")
print(f"The f1 score for knn on the training set is: {f1_score(y_train, knn_model.predict(X_train), average='macro')}")


# In[17]:


print(classification_report(y_val, predictions_knn))


# In[3]:


from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
device = torch.device("cuda")

classes = np.array([0, 1, 2]) 
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print("Class weights:", class_weights)


# In[4]:


import torch.nn as nn
import torch.nn.functional as F

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


# In[5]:


X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(device)
X_test_tensor = torch.tensor(X_val, dtype=torch.float).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_val.values, dtype=torch.long).to(device)


# In[6]:


model = BERTClassifier(input_dim=X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-4)


# In[7]:


from torch.utils.data import TensorDataset, DataLoader

# Create dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")


# In[8]:


model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    preds = torch.argmax(outputs, dim=1)

from sklearn.metrics import f1_score, classification_report
print("F1 Score (Macro):", f1_score(y_test_tensor.cpu(), preds.cpu(), average='macro'))
print(classification_report(y_test_tensor.cpu(), preds.cpu()))


# In[9]:


with torch.no_grad():
    outputs_train = model(X_train_tensor)
    train_preds = torch.argmax(outputs_train, dim=1)

from sklearn.metrics import f1_score, classification_report
print("F1 Score (Macro):", f1_score(y_train_tensor.cpu(), train_preds.cpu(), average='macro'))


# In[10]:


import torch
import os

save_path = r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\model\nn_model.pt"

# Ensure directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

torch.save(model.state_dict(), save_path)


# In[ ]:





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




