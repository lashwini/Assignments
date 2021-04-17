#!/usr/bin/env python
# coding: utf-8

# In[24]:


# Random Forest Classification
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[4]:


file = pd.read_csv('Company_Data (1).csv')
file


# In[7]:


label_encoder = preprocessing.LabelEncoder()
file['Sales']= label_encoder.fit_transform(file['Sales']) 
file


# In[10]:


file['Sales'].describe()


# In[11]:


file['Sales_New']=pd.cut(file.Sales,bins=[0,163,335],labels=['Low','High'])
df2=file.drop(['Sales'],axis=1)
df2


# In[12]:


df=pd.get_dummies(df2)


# In[13]:


df


# In[14]:


df1=df.drop(['Sales_New_High'],axis=1)
df1


# In[22]:


X = df1.drop(['Sales_New_Low'],axis=1)
y = df1['Sales_New_Low']


# In[25]:


num_trees = 100
max_features = 5
kfold = KFold(n_splits=10, random_state=7)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean()) 


# In[ ]:





# In[ ]:





# In[ ]:




