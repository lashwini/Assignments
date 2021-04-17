#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


book = pd.read_csv('book (2).csv', error_bad_lines = False,encoding="unicode_escape")
book.head()


# In[4]:


book1 = book.iloc[:,1:] 
book1


# In[6]:


book2 = book1.rename({'User.ID': 'userId','Book.Title':'Booktitle','Book.Rating':'Rating'},axis=1)
book2.head()


# In[7]:


book2.groupby('Booktitle')['Rating'].mean().head()


# In[8]:


book2.groupby('Booktitle')['Rating'].mean().sort_values(ascending=False).head()


# In[9]:


book2.groupby('Booktitle')['Rating'].count().sort_values(ascending=False).head()


# In[10]:


ratings_mean_count = pd.DataFrame(book2.groupby('Booktitle')['Rating'].mean())


# In[11]:


ratings_mean_count['rating_counts'] = pd.DataFrame(book2.groupby('Booktitle')['Rating'].count())


# In[12]:


ratings_mean_count.head()


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating_counts'].hist(bins=50)


# In[15]:


plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['Rating'].hist(bins=50)


# In[16]:


plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='Rating', y='rating_counts', data=ratings_mean_count, alpha=0.4)


# In[18]:


user_movie_rating = book2.pivot_table(index='userId', columns='Booktitle', values='Rating')
user_movie_rating


# In[30]:


user_movie_rating.fillna(0, inplace=True)
user_movie_rating


# In[36]:


## jason madison book has highest rating 


# In[31]:


Jason_Madison_amp_ratings = user_movie_rating[' Jason, Madison &amp']


# In[32]:


Jason_Madison_amp_ratings.head()


# In[33]:


book_like_jason_Madison = user_movie_rating.corrwith(Jason_Madison_amp_ratings)

corr_Jason_Madison = pd.DataFrame(book_like_jason_Madison, columns=['Correlation'])
corr_Jason_Madison.dropna(inplace=True)
corr_Jason_Madison.head()


# In[34]:


corr_Jason_Madison.sort_values('Correlation', ascending=False).head(10)


# In[ ]:




