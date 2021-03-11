#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[56]:


Data = pd.read_csv('delivery_time.csv')
Data.head(10)


# In[76]:


df = pd.DataFrame(Data)


# In[77]:


df.columns = df.columns.str.replace(' ', '') 
  


# In[78]:


df


# In[79]:


df.isna()


# In[80]:


df.describe()


# In[82]:


sns.distplot(df['DeliveryTime'])


# In[83]:


sns.distplot(df['SortingTime'])


# In[84]:


plt.plot(df['DeliveryTime'],df['SortingTime'],"bo")
plt.xlabel("DeliveryTime")
plt.ylabel("SortingTime")


# In[85]:


df['SortingTime'].corr(df['DeliveryTime'])


# In[87]:


import statsmodels.formula.api as smf
model = smf.ols("SortingTime~DeliveryTime",data=Data).fit()
model.params


# In[89]:


sns.regplot(x="DeliveryTime",y="SortingTime",data=df)


# In[91]:


model.summary()


# In[93]:


model.resid
model.resid_pearson


# In[95]:


print(model.conf_int(0.05))


# In[99]:


pred=model.predict(df.iloc[:,0])
pred
pd.set_option("display.max_rows",None)
rmse_lin = np.sqrt(np.mean((np.array(df['SortingTime'])-np.array(pred))**2))
rmse_lin 


# In[101]:


import matplotlib.pylab as plt
plt.scatter(x=df['DeliveryTime'],y=df['SortingTime'],color='red')
plt.plot(df['DeliveryTime'],pred,color='black')
plt.xlabel('DeliveryTime')
plt.ylabel('SortingTime')


# In[103]:


model2 = smf.ols('SortingTime~np.log(DeliveryTime)',data=df).fit()
model2.params
model2.resid 
model2.resid_pearson 


# In[105]:


model2.summary()


# In[108]:


pred2 = model2.predict(pd.DataFrame(df['DeliveryTime']))


# In[110]:


pred2
rmse_log = np.sqrt(np.mean((np.array(df['SortingTime'])-np.array(pred2))**2))
rmse_log 


# In[112]:


pred2.corr(df.SortingTime)


# In[116]:


plt.scatter(x=df['DeliveryTime'],y=df['SortingTime'],color='green')
plt.plot(df['DeliveryTime'],pred2,color='blue')
plt.xlabel('DeliveryTime')
plt.ylabel('SortingTime')


# In[118]:


model3 = smf.ols('np.log(SortingTime)~DeliveryTime',data=df).fit()
model3.params
model3.summary()


# In[120]:


pred_log = model3.predict(pd.DataFrame(df['DeliveryTime']))


# In[122]:


pred_log


# In[162]:


pred3=np.exp(pred_log)  # as we have used log(SortingTime) in preparing model so we need to convert it back
pred3 


# In[126]:


rmse_exp = np.sqrt(np.mean((np.array(df['SortingTime'])-np.array(pred3))**2))
rmse_exp  


# In[128]:


pred3.corr(df.SortingTime)


# In[130]:


plt.scatter(x=df['DeliveryTime'],y=df['SortingTime'],color='green')
plt.plot(df['DeliveryTime'],pred3,color='blue')
plt.xlabel('DeliveryTime')
plt.ylabel('SortingTime')


# In[132]:


student_resid = model3.resid_pearson 
student_resid 


# In[136]:


plt.plot(model3.resid_pearson,'o')
plt.axhline(y=0,color='green')
plt.xlabel("Observation Number")
plt.ylabel("Standardized Residual")


# In[137]:


plt.scatter(x=pred3,y=df.SortingTime)
plt.xlabel("Predicted")
plt.ylabel("Actual")


# In[139]:


df["DeliveryTime_Sq"] = df.DeliveryTime*df.DeliveryTime
df 


# In[140]:


model_quad = smf.ols("np.log(SortingTime)~DeliveryTime+DeliveryTime_Sq",data=df).fit()
model_quad.params


# In[142]:


model_quad.summary()


# In[145]:


pred_quad = model_quad.predict(df)
pred4=np.exp(pred_quad)  # as we have used log(SortingTime) in preparing model so we need to convert it back
pred4
rmse_quad = np.sqrt(np.mean((np.array(df['SortingTime'])-np.array(pred4))**2))
rmse_quad  


# In[151]:


plt.scatter(df.DeliveryTime,df.SortingTime,c="b")
plt.plot(df.DeliveryTime,pred4,"r") 


# In[152]:


plt.scatter(np.arange(21),model_quad.resid_pearson)
plt.axhline(y=0,color='red')
plt.xlabel("Observation Number")
plt.ylabel("Standardized Residual")


# In[154]:


data = {"MODEL":pd.Series(["rmse_lin","rmse_log","rmse_exp","rmse_quad"]),
        "RMSE_Values":pd.Series([rmse_lin,rmse_log,rmse_exp,rmse_quad]),
        "Rsquare":pd.Series([model.rsquared,model2.rsquared,model3.rsquared,model_quad.rsquared])}
table=pd.DataFrame(data)
table 


# In[156]:


print(plt.style.available) 


# In[158]:


import matplotlib.pyplot as plt
plt.style.use('dark_background')  


# In[160]:


plt.hist(model_quad.resid_pearson) 


# In[161]:


plt.scatter(df.DeliveryTime,df.SortingTime,c="b")
plt.plot(df.DeliveryTime,pred4,"r") 

