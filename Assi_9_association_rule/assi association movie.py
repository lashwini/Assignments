#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('my_movies (1).csv',sep=',')
dataset = dataset.replace(np.nan, '', regex=True)
dataset 


# In[2]:


len(dataset.columns)


# In[3]:


dataset
transactions = []
for i in range(0, 9):
    transactions.append([str(dataset.values[i,u]) for u in range(1, 14)]) 


# In[4]:


movie_series  = pd.DataFrame(pd.Series(transactions))
movie_series
movie_series.columns = ["business"]
movie_series


# In[5]:


X = movie_series['business'].str.join(sep=',').str.get_dummies(sep=',')
X


# In[6]:


get_ipython().system('pip install mlxtend')


# In[67]:


from mlxtend.frequent_patterns import apriori,association_rules
frequent_itemsets = apriori(X, min_support=0.004, max_len=3,use_colnames = True)

frequent_itemsets 


# In[65]:


frequent_itemsets.sort_values('support',ascending = False,inplace=True)
frequent_itemsets.sort_values


# In[68]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False).head(10)


# In[80]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[83]:


plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()


# In[85]:


fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
 fit_fn(rules['lift']))

