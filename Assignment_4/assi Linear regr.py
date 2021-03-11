#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# reading a csv file using pandas library
salary=pd.read_csv("Salary_Data.csv")
salary.head(10)


# In[35]:


salary.columns


# In[38]:


plt.boxplot(salary.YearsExperience)
salary.describe()


# In[18]:


salary


# In[24]:


sns.distplot(salary.YearsExperience)


# In[9]:


plt.boxplot(salary.Salary)


# In[25]:


sns.distplot(salary.Salary)


# In[11]:


plt.plot(salary.YearsExperience,salary.Salary,"bo")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")


# In[13]:


salary.Salary.corr(salary.YearsExperience)


# In[15]:


import statsmodels.formula.api as smf
model = smf.ols("Salary~YearsExperience",data = salary).fit()
model.params


# In[28]:


sns.regplot(x="YearsExperience",y="Salary",data=salary)


# In[17]:


model.summary()


# In[30]:


# t and p values
print(model.tvalues,"\n",model.pvalues)


# predict for new data points

# In[49]:


exp = pd.Series([3.2,4])


# In[50]:


salary_pred=pd.DataFrame(exp,columns=['YearsExperience'])


# In[51]:


model.predict(salary_pred)

