#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
data = pd.read_csv('book (1).csv',sep=',')
data = data.replace(np.nan, '', regex=True)
data 


# In[4]:


data
transactions = []
for i in range(0, 2000):
    transactions.append([str(data.values[i,u]) for u in range(1, 10)]) 


# In[5]:


book_series  = pd.DataFrame(pd.Series(transactions))
book_series
book_series.columns = ["business"]
book_series


# In[6]:


get_ipython().system('pip install mlxtend')


# In[7]:



from mlxtend.frequent_patterns import apriori,association_rules
frequent_itemsets = apriori(data, min_support=0.005, max_len=3,use_colnames = True)

frequent_itemsets 


# In[8]:


frequent_itemsets.sort_values('support',ascending = False,inplace=True)
frequent_itemsets.sort_values


# In[18]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False).head(5)


# In[20]:


import matplotlib.pyplot as plt
plt.scatter(rules['support'], rules['confidence'], alpha=0.9)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[21]:


plt.scatter(rules['support'], rules['lift'], alpha=0.9)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()


# In[22]:


fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
 fit_fn(rules['lift']))

