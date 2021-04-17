#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[2]:


data = pd.read_csv("forestfires (1).csv")
data
data.head()


# In[3]:


from sklearn.preprocessing import LabelEncoder


# In[4]:


labelencoder_data = LabelEncoder()
data=data.apply(LabelEncoder().fit_transform)


# In[5]:


data


# In[6]:


data.describe()


# In[7]:


data.isna()


# In[8]:


data.dtypes


# In[10]:


sns.histplot(x="rain",y="size_category",data=data,palette = "hls")


# In[12]:


sns.boxplot(x="wind",y="size_category",data=data,palette = "hls")


# In[13]:


sns.barplot(x='size_category',y='area',data=data,palette="hls")


# In[14]:


X = data.drop(['size_category'],axis=1)
y=data['size_category']


# In[15]:


X


# In[16]:


y


# In[ ]:





# In[17]:


X.shape


# In[18]:


y.shape


# In[19]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=7)


# In[28]:


model_linear = SVC(kernel = "linear")
model_linear.fit(X_train,y_train)
pred_test_linear = model_linear.predict(X_test)

np.mean(pred_test_linear==y_test)


# In[29]:


model_poly = SVC(kernel = "poly")
model_poly.fit(X_train,y_train)
pred_test_poly = model_poly.predict(X_test)
np.mean(pred_test_poly==y_test)


# In[30]:


# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(X_train,y_train)
pred_test_rbf = model_rbf.predict(X_test)

np.mean(pred_test_rbf==y_test)

