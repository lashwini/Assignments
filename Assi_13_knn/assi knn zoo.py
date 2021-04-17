#!/usr/bin/env python
# coding: utf-8

# In[1]:


# KNN Classification
import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[2]:


file = pd.read_csv('Zoo.csv')
file


# In[3]:


X=file.drop(['type','animal name'],axis=1).values
y=file['type'].values


# In[4]:


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)


# In[5]:


y_pred = knn.predict(X_test)


# In[6]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))


# In[7]:


print(confusion_matrix(y_test,y_pred))


# In[8]:


knn.score(X_train,y_train)


# In[9]:


knn.score(X_test,y_test)


# In[11]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# choose k between 1 to 41
k_range = range(1, 41)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=4)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[1]:


## for k value 0 to 5 accuracy is higher


# In[ ]:




