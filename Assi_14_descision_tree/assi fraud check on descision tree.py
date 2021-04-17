#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing 


# In[34]:


# import some data to play with
check = pd.read_csv('Fraud_check.csv')  


# In[35]:


df=pd.DataFrame(check)


# In[36]:


df


# In[37]:


df1=df[['Taxable.Income','Undergrad','Marital.Status','Work.Experience','Urban']]


# In[38]:


df1


# In[39]:


df1['Taxable.Income'].max()


# In[40]:


df1['Tax_New']=pd.cut(df1['Taxable.Income'],bins=[0,30000,99619+1],labels=['Risky','Good'])
df2=df1.drop(['Taxable.Income'],axis=1)
df2


# In[41]:


df3 = pd.get_dummies(df2)
df3


# In[42]:


df4=df3.drop(['Tax_New_Risky'],axis=1)
df4


# In[83]:


X=df4.drop(['Tax_New_Good'],axis=1)
Y=df4['Tax_New_Good']


# In[44]:


Y


# In[84]:


X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)


# ## using entropy criteria

# In[69]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=4)
model.fit(X_train,Y_train) 


# In[70]:


tree.plot_tree(model);


# In[71]:


fn=['Work.Experience','Undergrad_NO','Undergrad_YES','Marital.Status_Divorced','Marital.Status_Married','Marital.Status_Single','Urban_NO','Urban_YES']
cn=['Risky','Good']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,5), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True); 


# In[72]:


model.feature_importances_ 


# In[73]:


import pandas as pd
feature_imp = pd.Series(model.feature_importances_,index=fn).sort_values(ascending=False) 
feature_imp


# In[77]:


preds = model.predict(X_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category  


# In[78]:


preds


# In[85]:


pd.crosstab(Y_test,preds)  


# In[86]:


np.mean(preds==Y_test)


# ## Using gini criteria

# In[80]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3) 


# In[81]:


model_gini.fit(X_train, y_train) 


# In[26]:


#Prediction and computing the accuracy
pred=model.predict(X_test)
np.mean(preds==Y_test) 

