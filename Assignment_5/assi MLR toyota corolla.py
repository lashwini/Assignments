#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np


# In[3]:


cars = pd.read_csv("ToyotaCorolla.csv",encoding= 'unicode_escape')
cars.head()


# In[4]:


cars.info()


# In[5]:


cars.isna().sum()


# In[6]:


cars.columns


# In[11]:


cars.corr()


# In[8]:


import statsmodels.formula.api as smf 
model = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=cars).fit()


# In[11]:


model.params


# In[13]:


#t and p-Values
print(model.tvalues, '\n', model.pvalues)


# In[15]:


(model.rsquared,model.rsquared_adj)


# In[17]:


model.summary()


# ## Model prediction

# In[19]:


#New data for prediction
new_data=pd.DataFrame({'Age_08_04':28,'KM':41000,'HP':90,'cc':2000,'Doors':3,'Gears':5,'Quarterly_Tax':210,'Weight':1165},index=[1])


# In[21]:


model.predict(new_data)


# In[22]:


model.predict(cars.iloc[0:5,])

