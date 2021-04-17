#!/usr/bin/env python
# coding: utf-8

# ## Q1

# In[ ]:


##A F&B manager wants to determine whether there is any significant difference in the diameter of the cutlet between two units.
##A randomly selected sample of cutlets was collected from both units and measured? Analyze the data and draw inferences at 5% significance level. 
##Please state the assumptions and tests that you carried out to check validity of the assumptions.


# In[30]:


# define hypothesis
# H0 = diameter of the cutlets are equal
# H1 = diameter of the cutlets are not equal


# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy import stats
import random


# In[2]:


file = pd.read_csv('Cutlets.csv')


# In[3]:


file


# In[4]:


file.describe()


# In[5]:


sample1 = random.choices(file['Unit A'],k=10)


# In[6]:


sample1


# In[7]:


sample2 = random.choices(file['Unit B'],k=10)
sample2


# In[8]:


stats.ttest_ind(sample1,sample2)


# In[11]:


# here pvalue>0.05
# we accept null hypothesis
# diameter of the cutlets are same in both units


# ## Q2

# In[12]:


#A hospital wants to determine whether there is any difference in the average Turn Around Time (TAT) of reports of the laboratories on their preferred list. 
#They collected a random sample and recorded TAT for reports of 4 laboratories. TAT is defined as sample collected to report dispatch.
#  Analyze the data and determine whether there is any difference in average TAT among the different laboratories at 5% significance level.
#   Minitab File: LabTAT.mtw


# In[13]:


# define hypothesis
#H0 = there is no difference in avg tat
#H1 = at least one of the laboratories avg time is different


# In[9]:


import pandas as pd 
from scipy import stats


# In[10]:


lab = pd.read_csv('LabTAT.csv')


# In[11]:


lab


# In[12]:


lab.columns = "Laboratory1","Laboratory2","Laboratory3","Laboratory4"
stats.f_oneway(lab.iloc[:,0], lab.iloc[:,1], lab.iloc[:,2], lab.iloc[:,3])


# In[22]:


#pvalue>o.05
#we accept null hypothesis
# there is no difference in avg tat


# ## Q3

# In[23]:


#Sales of products in four different regions is tabulated for males and females.
#Find if male-female buyer rations are similar across regions.
#Sales of products in four different regions is tabulated for males and females. Find if male-female buyer rations are similar across regions.


# In[39]:


# H0-male female buyer ratios are equal
# H1-male female buyer ratios are not equal


# In[40]:


import pandas as pd
from scipy import stats


# In[42]:


sop = pd.read_csv('BuyerRatio.csv')
sop


# In[47]:


sop1 = pd.DataFrame(sop)


# In[51]:


sop1['all'] = sop1.sum(axis=1)


# In[52]:


sop1


# In[55]:


count=pd.crosstab(sop1['Observed Values'],sop1['all'])


# In[56]:


Chisquares_results=scipy.stats.chi2_contingency(count)
Chisquares_results

 


# In[57]:


chisquare_results=[['','Test Statistic','p-value'],['Sample Data',chisquare_results[0],chisquare_results[1]]]
chisquare_results 


# In[1]:


## p value is greater than 0.05 hence we accept null hypothesis
# male female buyer ratio are similar


# ## Q4a

# In[97]:


#TeleCall uses 4 centers around the globe to process customer order forms. They audit a certain %  of the customer order forms.
#Any error in order form renders it defective and has to be reworked before processing. 
#he manager wants to check whether the defective %  varies by centre. Please analyze the data at 5% significance level
#and help the manager draw appropriate inferences


# In[38]:


# H0- % of defective of all countries are equal
# H1- % of defective are not equal


# In[17]:


import pandas as pd
import scipy
from scipy import stats


# In[18]:


form = pd.read_csv('Costomer+OrderForm.csv')


# In[19]:


form


# In[20]:


from sklearn.preprocessing import LabelEncoder


# In[21]:


labelencoder_form1=LabelEncoder()
form=form.apply(LabelEncoder().fit_transform)


# In[22]:


form


# In[26]:


chisquare_results=scipy.stats.chi2_contingency(form)
chisquare_results


# In[35]:


chisquare_results=[['','Test Statistic','p-value'],['Sample Data',chisquare_results[0],chisquare_results[1]]]
chisquare_results 


# In[37]:


## p value is greater than 0.05, we accept null hypothesis


# ## Q4b

# In[ ]:


##Fantaloons Sales managers commented that % of males versus females walking in to the store differ based on day of the week.
##Analyze the data and determine whether there is evidence at 5 % significance level to support this hypothesis.
 


# In[36]:


# H0 = male and females are equal
# H1 = male and females are not equal


# In[10]:


import pandas as pd
import scipy
from scipy import stats


# In[11]:


sale = pd.read_csv('Faltoons.csv')
sale


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


labelencoder_sale=LabelEncoder()
sale=sale.apply(LabelEncoder().fit_transform)


# In[14]:


sale


# In[15]:


import statsmodels.api as sm
sm.stats.ttest_ind(sale.Weekdays,sale.Weekend)


# In[ ]:


## p value is greater than 0.05 we accept null hypothesis 


# In[ ]:





# In[ ]:




