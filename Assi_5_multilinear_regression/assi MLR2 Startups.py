#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.api as smf
import numpy as np
from sklearn import preprocessing


# In[2]:


startup = pd.read_csv('50_Startups.csv')
startup.head()


# In[3]:


startup.columns = startup.columns.str.replace(' ','')


# In[4]:


startup.head()


# In[5]:


startup.info()


# In[6]:


startup.isna().sum()


# In[7]:


startup.columns


# In[8]:


startup.rename(columns={'R&DSpend':'RDSpend'},inplace=True)


# In[10]:


startup.head()


# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[148]:


label_encoder = preprocessing.LabelEncoder()
startup["State"]=label_encoder.fit_transform(startup["State"])
startup.head()
st=startup
st


# In[149]:


from sklearn.preprocessing import LabelEncoder,StandardScaler
std_sclr = StandardScaler()
st = std_sclr.fit_transform(st)


# In[150]:


st = pd.DataFrame(st)
st.head()


# In[152]:


st.columns = ["RDSpend","Administration","MarketingSpend","State","Profit"]
st.head()


# In[153]:


st.corr()


# ## Scatterplot between variables along with histogram
# 

# In[154]:


sns.set_style(style='darkgrid')
sns.pairplot(st)


# ## Preparing a model

# In[155]:


# build model
import statsmodels.formula.api as smf
model = smf.ols('Profit~RDSpend+Administration+MarketingSpend+State' ,data=st).fit()


# In[156]:


model.params


# In[157]:


# t and p values
print(model.tvalues, '\n',model.pvalues)


# In[158]:


model.rsquared,model.rsquared_adj


# In[159]:


model.summary()


# In[160]:


import statsmodels.formula.api as smf
model1 = smf.ols(formula='Profit~RDSpend+MarketingSpend+State' ,data=st).fit()


# In[161]:


model1.params


# In[162]:


model1.summary()


# In[163]:


model2=smf.ols('Profit~RDSpend+Administration+MarketingSpend',data = st).fit()  
print(model2.tvalues, '\n', model2.pvalues)  


# In[164]:


model2.summary()


# In[165]:


rsq_RD = smf.ols('RDSpend~Administration+MarketingSpend',data=startup).fit().rsquared  
vif_RD = 1/(1-rsq_RD) 

rsq_Adm = smf.ols('Administration~RDSpend+MarketingSpend',data=startup).fit().rsquared  
vif_Adm = 1/(1-rsq_Adm)

rsq_MS = smf.ols('MarketingSpend~RDSpend+Administration',data=startup).fit().rsquared  
vif_MS = 1/(1-rsq_MS) 



# Storing vif values in a data frame
d1 = {'Variables':['MarketingSpend','RDSpend','Administration'],'VIF':[vif_RD,vif_Adm,vif_MS]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# In[166]:


import statsmodels.api as sm
qqplot=sm.qqplot(model2.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[167]:


list(np.where(model.resid>0.3))


# In[168]:


model2_influence = model2.get_influence()
(c, _) = model2_influence.cooks_distance


# In[169]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(st)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[170]:


(np.argmax(c),np.max(c))


# In[171]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model2)
plt.show()


# In[172]:


k = startup.shape[1]
n = startup.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# In[173]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[94]:


plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[174]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model2, "RDSpend", fig=fig)
plt.show()


# In[97]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model2, "Administration", fig=fig)
plt.show()


# In[99]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model2, "MarketingSpend", fig=fig)
plt.show()


# ## cooks distance

# In[175]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


# In[176]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(st)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[177]:


(np.argmax(c),np.max(c))


# In[178]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model2)
plt.show()


# In[179]:


k = st.shape[1]
n = st.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# In[183]:


st[st.index.isin([49])]


# In[184]:


st.head()


# In[132]:


startup_new = startup


# In[133]:


st2=startup_new.drop(st_new.index[[49]],axis=0).reset_index()
st2 


# In[134]:


st2 = st2.drop(['index'],axis=1)


# In[135]:


st2


# In[136]:


final_ml_A= smf.ols('Profit~RDSpend+Administration+MarketingSpend',data=st2).fit()


# In[137]:


(final_ml_A.rsquared,final_ml_A.aic)


# In[138]:


#Again check for influencers
model_influence_A = final_ml_A.get_influence()
(c_V, _) = model_influence_A.cooks_distance


# In[139]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(st2)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[140]:


(np.argmax(c_V),np.max(c_V))


# Since the value is <1 , we can stop the diagnostic process and finalize the model
# 

# In[141]:


final_ml_A= smf.ols('Profit~RDSpend+Administration+MarketingSpend',data=st2).fit()


# In[142]:


(final_ml_A.rsquared,final_ml_A.aic)


# ## predicting for new data

# In[143]:


new_data=pd.DataFrame({'RDSpend':162597.70,"Administration":151377.59,"MarketingSpend":443898.53},index=[1])


# In[144]:


model2.predict(new_data)


# In[ ]:




