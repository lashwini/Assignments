#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
crime = pd.read_csv("crime_data.csv")
crime


# In[33]:


from sklearn.preprocessing import MinMaxScaler
trans = MinMaxScaler()
data = pd.DataFrame(trans.fit_transform(crime.iloc[:,1:]))
data 


# In[4]:


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


# In[34]:


from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=5, linkage='complete',affinity = "euclidean").fit(data) 

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime


# In[35]:


crime.iloc[:,1:].groupby(crime.clust).mean()


# ## Kmeans clustering

# In[37]:


from sklearn.cluster import KMeans
import numpy as np


# In[42]:


crime1=pd.read_csv('crime_data.csv')
crime1.head()


# In[44]:


X = np.random.uniform(0,1,1000)
Y = np.random.uniform(0,1,1000)
X


# In[46]:


df_xy =pd.DataFrame(columns=["X","Y"])
df_xy
df_xy.X = X
df_xy.Y = Y
df_xy
df_xy.plot(x="X",y = "Y",kind="scatter")
model1 = KMeans(n_clusters=5).fit(df_xy)


# In[48]:


df_xy.plot(x="X",y = "Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm_r)


# In[50]:


def norm_func(i):
    x = (i-i.min()) / (i.max() - i.min())
    return (x)


# In[54]:


df_norm = norm_func(crime1.iloc[:,1:])


# In[56]:


df_norm.head(10) 


# In[58]:


from sklearn.cluster import KMeans
fig = plt.figure(figsize=(10, 8))
WCSS = []
for i in range(1, 11):
    clf = KMeans(n_clusters=i)
    clf.fit(df_norm)
    WCSS.append(clf.inertia_) # inertia is another name for WCSS
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()


# In[60]:


clf = KMeans(n_clusters=5)
y_kmeans = clf.fit_predict(df_norm)


# In[62]:


y_kmeans
#clf.cluster_centers_
clf.labels_


# In[84]:


md=pd.Series(y_kmeans)  # converting numpy array into pandas series object 
crime1['clust']=md # creating a  new column and assigning it to new column 
crime1


# In[82]:


crime1.iloc[:,0:5].groupby(crime1.clust).mean()


# In[69]:


crime1.plot(x="Rape",y ="Murder",c=clf.labels_,kind="scatter",s=50 ,cmap=plt.cm.coolwarm) 
plt.title('Clusters using KMeans')


# ## DBScan

# In[90]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
cr1=pd.read_csv('crime_data.csv')
cr1.head()


# In[96]:


array=cr1.iloc[:,1:5].values
array  


# In[98]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array) 
X  


# In[162]:


dbscan = DBSCAN(eps=1.3, min_samples=5)
dbscan.fit(X)   


# In[163]:


#Noisy samples are given the label -1.
dbscan.labels_    


# In[165]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster']) 


# In[166]:


cl
pd.set_option("display.max_rows", None) 


# In[167]:


cl


# In[169]:


df1 = pd.concat([cr1,cl],axis=1) 
df1  


# In[171]:


import matplotlib.pyplot as plt
plt.style.use('classic') 


# In[173]:


cr1.plot(x="Murder",y ="UrbanPop",c=dbscan.labels_ ,kind="scatter",s=50 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using DBScan')


# In[174]:


##using hclustering and kmeans clusters are same = 5
## using dbscan number of clusters = 2


# In[ ]:




