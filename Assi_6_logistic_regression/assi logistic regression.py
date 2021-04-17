#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report


# In[44]:


#Loading bank data
bank = pd.read_csv("bank-full.csv",delimiter=';')
bank.head()


# In[35]:


bank.isnull().sum()


# In[48]:


labelencoder_bank = LabelEncoder()
bank=bank.apply(LabelEncoder().fit_transform)


# In[49]:


c1=bank
c1.head()


# In[50]:


c1.columns


# In[51]:


c1.describe()


# In[27]:


cor_mat=c1.corr()
fig = plt.Figure(figsize=(15,7))
sns.heatmap(cor_mat,annot=True)


# In[52]:


c1.drop(['default'],inplace=True,axis = 1)


# In[53]:


c1.head()


# In[ ]:





# In[25]:


sb.countplot(x="y",data=bank,palette="hls")


# In[28]:


pd.crosstab(bank.y,bank.loan) 


# In[30]:


pd.crosstab(bank.y,bank.loan).plot(kind = 'bar') 


# In[34]:


pd.crosstab(bank.y,bank.housing).plot(kind = 'bar') 


# In[54]:


# Model building 
from sklearn.linear_model import LogisticRegression
bank.shape  


# In[74]:


X = c1.drop(['y'],axis=1)
Y = c1['y']
classifier = LogisticRegression()
classifier.fit(X,Y)


# In[75]:


classifier.coef_  


# In[76]:


classifier.predict_proba (X) 


# In[77]:


y_pred = classifier.predict(X)
c1["y_pred"] = y_pred
c1  


# In[78]:


y_prob = pd.DataFrame(classifier.predict_proba(X))
new_df = pd.concat([c1,y_prob],axis=1)
new_df 


# In[79]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)


# In[81]:


pd.crosstab(y_pred,Y) 


# In[84]:


#type(y_pred)
accuracy = sum(Y==y_pred)/c1.shape[0]
accuracy


# In[86]:


from sklearn.metrics import classification_report 
print (classification_report (Y, y_pred)) 


# In[88]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
Logit_roc_score=roc_auc_score(Y,classifier.predict(X))
Logit_roc_score 


# In[90]:


fpr, tpr, thresholds = roc_curve(Y,classifier.predict_proba(X)[:,1]) 
plt.plot(fpr, tpr, label='Logistic Regression (area=%0.2f)'% Logit_roc_score)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()  


# In[92]:


y_prob1 = pd.DataFrame(classifier.predict_proba(X)[:,1])


# In[94]:


y_prob1 


# In[96]:


import statsmodels.api as sm  


# In[98]:


logit = sm.Logit(Y, X)  


# In[100]:


logit.fit().summary() 


# In[102]:


fpr 


# In[103]:


tpr

