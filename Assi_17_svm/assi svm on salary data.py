#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns


# In[2]:


test = pd.read_csv('SalaryData_Test.csv')
test


# In[3]:


test.shape


# In[4]:


test.isna().sum()


# In[5]:


test.loc[test['Salary'] == ' >50K', 'Salary'] = 1
test.loc[test['Salary'] == ' <=50K', 'Salary'] = 0


# In[6]:


test


# In[8]:


from sklearn.preprocessing import LabelEncoder


# In[9]:


labelencoder_test = LabelEncoder()
test=test.apply(LabelEncoder().fit_transform)


# In[10]:


X_test = test.drop(['Salary'],axis = 1)
Y_test = test['Salary']


# In[11]:


X_test


# In[12]:


train=pd.read_csv('SalaryData_Train.csv')
train


# In[13]:


train.shape


# In[14]:


train.isna().sum()


# In[15]:


train.loc[train['Salary'] == ' >50K', 'Salary'] = 1
train.loc[train['Salary'] == ' <=50K', 'Salary'] = 0


# In[16]:


train


# In[17]:


labelencoder_train = LabelEncoder()
train=train.apply(LabelEncoder().fit_transform)


# In[18]:


train


# In[19]:


X_train = train.drop(['Salary'],axis=1)
Y_train = train['Salary']


# In[20]:


X_train


# In[27]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[25]:


model_linear = SVC(kernel = "linear")
model_linear.fit(X_train,Y_train)
pred_test_linear = model_linear.predict(X_test)

np.mean(pred_test_linear==Y_test)


# In[26]:


model_poly = SVC(kernel = "poly")
model_poly.fit(X_train,Y_train)
pred_test_poly = model_poly.predict(X_test)
np.mean(pred_test_poly==Y_test) 


# In[28]:


# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(X_train,Y_train)
pred_test_rbf = model_rbf.predict(X_test)

np.mean(pred_test_rbf==Y_test)


# In[ ]:





# In[ ]:




