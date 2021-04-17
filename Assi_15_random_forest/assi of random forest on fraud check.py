#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Random Forest Classification
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[3]:


file = pd.read_csv('Fraud_check.csv')
file.head()


# In[4]:


file['Taxable.Income'].max()


# In[5]:


file['Tax_New']=pd.cut(file['Taxable.Income'],bins=[0,30000,99619+1],labels=['Risky','Good'])
df=file.drop(['Taxable.Income'],axis=1)
df


# In[6]:


df2 = pd.get_dummies(df)
df2


# In[7]:


df3=df2.drop(['Tax_New_Risky'],axis=1)
df3


# In[9]:


X=df3.drop(['Tax_New_Good'],axis=1)
Y=df3['Tax_New_Good']


# In[12]:


num_trees = 100
max_features = 5
kfold = KFold(n_splits=10, random_state=7)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean()) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




