#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, CategoricalNB


# In[9]:


test = pd.read_csv('SalaryData_Test.csv')
test


# In[10]:


test.shape


# In[11]:


test.isna().sum()


# In[19]:


test.loc[test['Salary'] == ' >50K', 'Salary'] = 1
test.loc[test['Salary'] == ' <=50K', 'Salary'] = 0


# In[20]:


test


# In[43]:


labelencoder_test = LabelEncoder()
test=test.apply(LabelEncoder().fit_transform)


# In[ ]:





# In[44]:


X_test = test.drop(['Salary'],axis = 1)
Y_test = test['Salary']


# In[45]:


X_test


# In[26]:


train=pd.read_csv('SalaryData_Train.csv')
train


# In[27]:


train.shape


# In[28]:


train.isna().sum()


# In[29]:


train.loc[train['Salary'] == ' >50K', 'Salary'] = 1
train.loc[train['Salary'] == ' <=50K', 'Salary'] = 0


# In[30]:


train


# In[39]:


labelencoder_train = LabelEncoder()
train=train.apply(LabelEncoder().fit_transform)


# In[40]:


train


# In[41]:


X_train = train.drop(['Salary'],axis=1)
Y_train = train['Salary']


# In[33]:


X_train


# In[1]:


## model


# In[53]:


gnb = GaussianNB()
gnb.fit(X_train, Y_train)


# In[58]:


Y_pred = gnb.predict(X_test)


# In[59]:


from sklearn import metrics


# In[60]:


print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

