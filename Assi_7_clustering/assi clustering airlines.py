#!/usr/bin/env python
# coding: utf-8

# ## Hclustering

# In[44]:


#Import the libraries
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   


# In[45]:


air = pd.read_excel(open('airlines.xlsx', 'rb'), sheet_name='data')
air.head() 


# In[46]:


air.columns


# In[47]:


air.isnull().sum()


# In[48]:


from sklearn.preprocessing import MinMaxScaler
trans = MinMaxScaler()
data = pd.DataFrame(trans.fit_transform(air.iloc[:,1:]))
data 


# In[6]:


from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 
#p = np.array(df_norm) # converting into numpy array format 
z = linkage(data, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
    )
plt.show()


# In[43]:


## it is difficult to cut the dendrogram


# ## Using Kmeans Clustering

# In[11]:


import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
import numpy as np


# In[29]:


X = np.random.uniform(0,1,1000)
Y = np.random.uniform(0,1,1000)
X


# In[30]:


df_xy =pd.DataFrame(columns=["X","Y"])
df_xy
df_xy.X = X
df_xy.Y = Y
df_xy
df_xy.plot(x="X",y = "Y",kind="scatter")
model1 = KMeans(n_clusters=3).fit(df_xy)


# In[31]:


df_xy.plot(x="X",y = "Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm_r)


# In[32]:


from sklearn.cluster import KMeans
fig = plt.figure(figsize=(10, 8))
WCSS = []
for i in range(1, 11):
    clf = KMeans(n_clusters=i)
    clf.fit(air)
    WCSS.append(clf.inertia_) # inertia is another name for WCSS
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()


# In[33]:


clf = KMeans(n_clusters=3)
y_kmeans = clf.fit_predict(data)


# In[34]:


y_kmeans
#clf.cluster_centers_
clf.labels_


# In[35]:


md=pd.Series(y_kmeans)  # converting numpy array into pandas series object 
air['clust']=md # creating a  new column and assigning it to new column 
air


# In[26]:


air.iloc[:,0:11].groupby(air.clust).mean()


# In[28]:


air.plot(x='ID#',y='cc1_miles',c=clf.labels_,kind="scatter",s=50 ,cmap=plt.cm.coolwarm) 
plt.title('Clusters using KMeans')


# ## DBSCAN 

# In[37]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   


# In[11]:


airline=pd.read_excel(open('airlines.xlsx', 'rb'), sheet_name='data')
airline.head()


# In[10]:


airline.info()


# In[13]:


array=airline.values
array  


# In[15]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array) 
X  


# In[17]:


dbscan = DBSCAN(eps=0.45, min_samples=5)
dbscan.fit(X)                


# In[19]:


dbscan.labels_  


# In[21]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])


# In[23]:


cl
pd.set_option("display.max_rows", None) 


# In[25]:


cl


# In[28]:


df1 = pd.concat([airline,cl],axis=1) 
df1     


# In[30]:


import matplotlib.pyplot as plt
plt.style.use('classic') 


# In[41]:


df1.plot(x="cc1_miles",y ="Bonus_miles",c=dbscan.labels_ ,kind="scatter",s=50 ,cmap=plt.cm.coolwarm) 
plt.title('Clusters using DBScan')   


# In[49]:


## using hclustering it is difficult to cut the dendrogram
## using kmeans no. of clusters =3
## using dbscan no of clusters =2


# In[ ]:




