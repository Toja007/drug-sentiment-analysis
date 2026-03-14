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

X_test = joblib.load(r'C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\X_val.pkl')
y_test = joblib.load(r'C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed\y_val.pkl')


# In[3]:


# train and evaluate logistic regression model as baseline
lr = LogisticRegression(max_iter=100)

lr.fit(X_train, y_train)


# In[4]:


#make predictiona on test set
predictions = lr.predict(X_test)

#print f1 score on test set
print(f"the f1 score for logistic regression on the test set is: {f1_score(y_test, predictions, average='macro')}")


# In[5]:


#f1 score on training set
print(f"The f1 score for logistic regression on the training set is: {f1_score(y_train, lr.predict(X_train), average='macro')}")


# In[6]:


print(classification_report(y_test, predictions))


# In[7]:


# train and evaluate decision tree model
dt = DecisionTreeClassifier(
    max_depth=20,
    random_state=42
)

dt.fit(X_train, y_train)


# In[8]:


#make prediction on test set with decision tree
predictions_dt = dt.predict(X_test)

#print f1 score on test set
print(f"the f1 score for decision tree on the test set is: {f1_score(y_test, predictions_dt, average='macro')}")
print(f"The f1 score for decision tree on the training set is: {f1_score(y_train, dt.predict(X_train), average='macro')}")


# In[9]:


print(classification_report(y_test, predictions_dt))


# In[ ]:


# train random forest model
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


# In[11]:


#make prediction on test set with random forest
predictions_rf = rf.predict(X_test)

#print f1 score on test set
print(f"the f1 score for random forest on the test set is: {f1_score(y_test, predictions_rf, average='macro')}")
print(f"The f1 score for random forest on the training set is: {f1_score(y_train, rf.predict(X_train), average='macro')}")


# In[12]:


print(classification_report(y_test, predictions_dt))


# In[13]:


# train and evaluate support vector machine model
svm = SVC()

svm.fit(X_train, y_train)


# In[14]:


#make prediction on test set with support vector machine
predictions_svm = svm.predict(X_test)

#print f1 score on test set
print(f"the f1 score for support vector machine on the test set is: {f1_score(y_test, predictions_svm, average='macro')}")
print(f"The f1 score for support vector machine on the training set is: {f1_score(y_train, svm.predict(X_train), average='macro')}")


# In[15]:


# train and evaluate Xgboost model
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)


# In[16]:


#make prediction on test set with Xgboost
predictions_xgb = xgb.predict(X_test)

#print f1 score on test set
print(f"the f1 score for Xgboost on the test set is: {f1_score(y_test, predictions_xgb, average='macro')}")
print(f"The f1 score for Xgboost on the training set is: {f1_score(y_train, xgb.predict(X_train), average='macro')}")


# In[ ]:


# train and evaluate logistic regression with cross validation
log_reg_cv = LogisticRegressionCV(
    cv=5,
    max_iter=1000,
    n_jobs=-1
)

log_reg_cv.fit(X_train, y_train)


# In[18]:


#make prediction on test set with Logistic Regression CV
predictions_lr_cv = log_reg_cv.predict(X_test)

#print f1 score on test set
print(f"the f1 score for Logistic Regression CV on the test set is: {f1_score(y_test, predictions_lr_cv, average='macro')}")
print(f"The f1 score for Logistic Regression CV on the training set is: {f1_score(y_train, log_reg_cv.predict(X_train), average='macro')}")


# In[19]:


# train and evaluate logistic regression model as baseline
lr2 = LogisticRegression(max_iter=100,
                         class_weight='balanced',
    C=2)

lr2.fit(X_train, y_train)


# In[20]:


#make prediction on test set with Logistic Regression CV 2
predictions_lr2 = lr2.predict(X_test)

#print f1 score on test set
print(f"the f1 score for Logistic Regression CV 2 on the test set is: {f1_score(y_test, predictions_lr2, average='macro')}")
print(f"The f1 score for Logistic Regression CV 2 on the training set is: {f1_score(y_train, lr2.predict(X_train), average='macro')}")


# In[21]:


#tuning decision tree with random ssearch cv
dt_rs = DecisionTreeClassifier(random_state=42)

dt_params = {
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy']
}

dt_search = RandomizedSearchCV(
    dt_rs,
    dt_params,
    n_iter=20,
    scoring='f1_weighted',
    cv=5,
    n_jobs=-1,
    verbose=1
)

dt_search.fit(X_train, y_train)

best_dt = dt_search.best_estimator_


# In[ ]:


# prediction an evaluation of the tuned decision tree model
from sklearn.metrics import f1_score

y_pred_dt_rs = best_dt.predict(X_test)

print("Tuned Decision Tree F1 on test:",
      f1_score(y_test, y_pred_dt_rs, average='macro'))

print("Tuned Decision Tree F1 on train:",
      f1_score(y_train, best_dt.predict(X_train), average='macro'))


# In[23]:


# tuning random forest with random search cv


rf_rs = RandomForestClassifier(random_state=42)

rf_params = {
    'n_estimators': [200, 300, 400],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_search = RandomizedSearchCV(
    rf_rs,
    rf_params,
    n_iter=20,
    scoring='f1_weighted',
    cv=5,
    n_jobs=-1,
    verbose=1
)

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_


# In[24]:


#make prediction on test set with Random Forest Random Search
y_pred_rf_rs = best_rf.predict(X_test)

#print f1 score on test set
print(f"the f1 score for Random Forest Random Search on the test set is: {f1_score(y_test, y_pred_rf_rs, average='macro')}")
print(f"The f1 score for Random Forest Random Search on the training set is: {f1_score(y_train, best_rf.predict(X_train), average='macro')}")


# In[25]:


# tuning support vector machine with random search cv
svm_rs = LinearSVC()

svm_params = {
    'C': [0.01, 0.1, 1, 5, 10, 20]
}

svm_search = RandomizedSearchCV(
    svm_rs,
    svm_params,
    n_iter=10,
    scoring='f1_weighted',
    cv=5,
    n_jobs=-1
)

svm_search.fit(X_train, y_train)

best_svm = svm_search.best_estimator_


# In[26]:


#make prediction on test set with Support Vector Machine Random Search
y_pred_svc_rs = best_svm.predict(X_test)

#print f1 score on test set
print(f"the f1 score for Support Vector Machine Random Search on the test set is: {f1_score(y_test, y_pred_svc_rs, average='macro')}")
print(f"The f1 score for Support Vector Machine Random Search on the training set is: {f1_score(y_train, best_svm.predict(X_train), average='macro')}")


# In[27]:


# tuning xgboost with random search cv
xgb_rs = XGBClassifier(random_state=42)

xgb_params = {
    'n_estimators': [200, 300, 400],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1],
    'colsample_bytree': [0.7, 0.8, 1]
}

xgb_search = RandomizedSearchCV(
    xgb_rs,
    xgb_params,
    n_iter=20,
    scoring='f1_weighted',
    cv=5,
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train, y_train)

best_xgb = xgb_search.best_estimator_


# In[28]:


#make prediction on test set with XGBoost Random Search
y_pred_xgb_rs = best_xgb.predict(X_test)

#print f1 score on test set
print(f"the f1 score for XGBoost Random Search on the test set is: {f1_score(y_test, y_pred_xgb_rs, average='macro')}")
print(f"The f1 score for XGBoost Random Search on the training set is: {f1_score(y_train, best_xgb.predict(X_train), average='macro')}")


# # model evalution
# 
# the f1 score for logistic regression on the test set is: 0.5930483635966732
# The f1 score for logistic regression on the training set is: 0.7116027125443153
# 
# the f1 score for decision tree on the test set is: 0.6338391290263241
# The f1 score for decision tree on the training set is: 0.8471768635624207
# 
# the f1 score for random forest on the test set is: 0.7322322451484325
# The f1 score for random forest on the training set is: 0.9229690559212544
# 
# the f1 score for support vector machine on the test set is: 0.41740576498485193
# The f1 score for support vector machine on the training set is: 0.4248719509003645
# 
# the f1 score for Xgboost on the test set is: 0.7497920595213582
# The f1 score for Xgboost on the training set is: 1.0
# 
# the f1 score for Logistic Regression CV on the test set is: 0.7376109072219701
# The f1 score for Logistic Regression CV on the training set is: 0.998611254811264
# 
# the f1 score for Logistic Regression CV 2 on the test set is: 0.6134098179211165
# The f1 score for Logistic Regression CV 2 on the training set is: 0.729689270487634
# 
# Tuned Decision Tree F1 on test: 0.709298489192205
# Tuned Decision Tree F1 on train: 1.0
# 
# the f1 score for Random Forest Random Search on the test set is: 0.7717761454325182
# The f1 score for Random Forest Random Search on the training set is: 0.9991661442853766
# 
# the f1 score for Support Vector Machine Random Search on the test set is: 0.7230218683362825
# The f1 score for Support Vector Machine Random Search on the training set is: 0.9745916304793285
# 
# the f1 score for XGBoost Random Search on the test set is: 0.7534195090813144
# The f1 score for XGBoost Random Search on the training set is: 0.9961078431450252
# 

# In[ ]:


# load the test set for final prediction
tt = joblib.load(r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\data\processed/X_test.pkl")


# In[ ]:


# make final prediction with the best random forst model
import numpy as np
np.unique(best_rf.predict(tt), return_counts=True)


# In[ ]:


# make final prediction with the best xgboost model
np.unique(best_xgb.predict(tt), return_counts=True)


# In[44]:


print(classification_report(y_test, y_pred_rf_rs))


# In[45]:


print(classification_report(y_test, y_pred_xgb_rs))


# # Best classical classifcation 
# the best model so far  is the *best_rf*

# In[47]:


joblib.dump(best_rf, r"C:\Users\GIGABYTE\Desktop\drug-sentiment-analysis\model\best_rf.joblib")


# In[ ]:




