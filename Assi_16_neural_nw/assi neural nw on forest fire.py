#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')


# In[45]:


from keras.models import Sequential
from keras.layers import Dense
import numpy 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[4]:


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy
import pandas as pd


# In[5]:


dataset = pd.read_csv('forestfires (1).csv')
dataset


# In[8]:


from sklearn.preprocessing import LabelEncoder


# In[9]:


labelencoder_dataset = LabelEncoder()
dataset=dataset.apply(LabelEncoder().fit_transform)


# In[10]:


dataset


# In[16]:


data = dataset.values
data


# In[24]:


import numpy as np


# In[25]:


X =  data[:,0:-1]
y = data[:,-1]


# In[29]:


X


# In[30]:


y


# In[31]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.33)


# In[38]:


model = Sequential()
model.add(Dense(50, input_dim=30, kernel_initializer='random_uniform', activation='relu'))
model.add(Dense(30, kernel_initializer='random_uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='random_uniform', activation='sigmoid')) 


# In[40]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# In[83]:


hist = model.fit(X, y, validation_split=0.33, epochs=150, batch_size=10) 


# In[85]:


scores = model.evaluate(X, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) 


# In[93]:


hist.history.keys()


# In[94]:


plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[95]:


# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss') 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

