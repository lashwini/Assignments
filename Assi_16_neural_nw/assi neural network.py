#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')


# In[2]:


# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy 
from sklearn.model_selection import train_test_split


# In[3]:


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy
import pandas as pd


# In[4]:


dataset = pd.read_csv('gas_turbines (1).csv')
dataset


# In[5]:


dataset.columns = ['AT','AP','AH','AFDP','GTEP','TIT','TAT','TEY','CDP','CO','NOX']
dataset


# In[6]:


data1=dataset[['AT','AP','AH','AFDP','GTEP','TIT','TAT','CDP','CO','NOX','TEY']]
data1


# In[7]:


from sklearn.preprocessing import LabelEncoder


# In[8]:


labelencoder_data1 = LabelEncoder()
data1=data1.apply(LabelEncoder().fit_transform)
data1


# In[9]:


data1['TEY'].describe()


# In[10]:


data1['TEY_New']=pd.cut(data1.TEY,bins=[0,1739,4206],labels=[0,1])
data2=data1.drop(['TEY'],axis=1)
data2


# In[11]:


data3 = data2.values


# In[12]:


data3


# In[13]:


X = data3[:,0:10]
y = data3[:,10]


# In[14]:


y


# In[15]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.33)


# In[16]:


model = Sequential()
model.add(Dense(32, input_dim=10, kernel_initializer='random_uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='random_uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='random_uniform', activation='sigmoid')) 


# In[17]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 


# In[18]:


# Fit the model
hist = model.fit(X, y,validation_split=0.33,epochs=150, batch_size=10) 


# In[19]:


scores = model.evaluate(X, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) 


# In[20]:


hist.history.keys()


# In[22]:


import matplotlib.pyplot as plt


# In[23]:


plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss') 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

