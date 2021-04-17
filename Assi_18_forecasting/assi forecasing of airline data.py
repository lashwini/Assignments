#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


air = pd.read_excel("Airlines+Data (1).xlsx")
air


# In[37]:


air.Passengers.plot()


# In[3]:


air['month'] = pd.DatetimeIndex(air['Month']).month
air.head()
pd.DataFrame(air)


# In[4]:


air['month_sqr'] = air[['month']].sum(axis=1).apply(np.square)
air


# In[5]:


air['log_pass'] = air[['Passengers']].sum(axis=1).apply(np.log)
air


# In[6]:


air1 = pd.get_dummies(air['month'])
pd.DataFrame(air1)


# In[7]:


air1.columns = ["jan","feb","march","april","may","june","july","august","sep","oct","nov","dec"]
air1.head()


# In[8]:


data = pd.concat([air,air1],axis=1)
data1 = pd.DataFrame(data)
data1


# ## Splitting the data

# In[9]:


Train = data1.head(84)
Test = data1.tail(12)


# In[10]:


#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~month',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['month'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear


# In[11]:


#Exponential

Exp = smf.ols('log_pass~month',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['month'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[12]:


#Quadratic 

Quad = smf.ols('Passengers~month+month_sqr',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["month","month_sqr"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad


# In[24]:


#Additive seasonality 

add_sea = smf.ols('Passengers~jan+feb+march+april+may+june+july+august+sep+oct+nov+dec',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['jan','feb','march','april','may','june','july','august','sep','oct','nov','dec']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[33]:


#Additive Seasonality Quadratic 

add_sea_Quad = smf.ols('Passengers~month+month_sqr+jan+feb+march+april+may+june+july+august+sep+oct+nov+dec',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['jan','feb','march','april','may','june','july','august','sep','oct','nov','dec','month','month_sqr']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# In[34]:


##Multiplicative Seasonality

Mul_sea = smf.ols('log_pass~jan+feb+march+april+may+june+july+august+sep+oct+nov+dec',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[36]:


#Compare the results 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# In[39]:


## rmse value for model additive seasonality is lower than other model.
## therefore model add_sea used for forecasing


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




