#!/usr/bin/env python
# coding: utf-8

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




