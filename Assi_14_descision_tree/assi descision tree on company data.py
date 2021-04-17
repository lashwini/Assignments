#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing 


# In[3]:


# import some data 
comp = pd.read_csv('Company_Data (1).csv')  


# In[4]:


comp.head()


# In[5]:


df=pd.DataFrame(comp)


# In[6]:


df1=df[['Sales','ShelveLoc','CompPrice','Income','Advertising','Population','Price','Age','Education','Urban','US']]


# In[8]:


df1


# In[9]:


label_encoder = preprocessing.LabelEncoder()
df1['Sales']= label_encoder.fit_transform(df1['Sales']) 


# In[10]:


df1['Sales']
df1['Sales'].describe()


# In[11]:


df1['Sales_New']=pd.cut(df1.Sales,bins=[0,163,335],labels=['Low','High'])
df2=df1.drop(['Sales'],axis=1)
df2


# In[14]:


df3=pd.get_dummies(df2)
df3


# In[16]:


df4=df3.drop(['Sales_New_High'],axis=1)
df4


# In[49]:


X = df4.drop(['Sales_New_Low'],axis=1)
y = df4['Sales_New_Low']


# In[50]:


X


# In[52]:


y


# In[54]:


# Splitting data into training and testing data set
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=45)


# In[86]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=4)
model.fit(X_train,y_train) 


# In[87]:


#PLot the decision tree
tree.plot_tree(model);


# In[88]:


fn=['ShalveLoc_Bad','ShelvLoc_Good','ShelveLoc_Medium','CompPrice','Income','Advertising','Population','Price','Age','Education','Urban_No','Urban_Yes','US_No','US_Yes']
cn=['Low','High']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,5), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True); 


# In[89]:


model.feature_importances_ 


# In[90]:


import pandas as pd
feature_imp = pd.Series(model.feature_importances_,index=fn).sort_values(ascending=False) 
feature_imp


# In[91]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()


# In[92]:


preds = model.predict(X_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category  


# In[93]:


preds


# In[94]:


pd.crosstab(y_test,preds)  


# In[96]:


np.mean(preds==y_test)


# ## Descision tree classifier using gini criteria

# In[80]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3) 


# In[81]:


model_gini.fit(X_train, y_train) 


# In[82]:


#Prediction and computing the accuracy
pred=model.predict(X_test)
np.mean(pred==y_test) 

