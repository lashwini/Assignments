#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


# In[2]:


wine = pd.read_csv("wine.csv")
wine.describe()
wine.head()


# In[3]:


wine.data = wine.iloc[:,1:]
wine.data.head()
# Converting into numpy array
WINE = wine.data.values
WINE


# In[5]:


wine_normal = scale(WINE)


# In[6]:


wine_normal


# In[7]:


pca = PCA(n_components = 13)
pca_values = pca.fit_transform(wine_normal)
pca_values 


# In[8]:


pca.components_


# In[9]:


var = pca.explained_variance_ratio_
var


# In[10]:


var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1


# In[11]:


plt.plot(var1,color="red")


# In[12]:


pca_values[:,0:1]


# In[13]:


finalDf = pd.concat([pd.DataFrame(pca_values[:,0:3],columns=['pc1','pc2','pc3']), wine[['Type']]], axis = 1)
finalDf


# In[14]:


import matplotlib.pyplot as plt
plt.style.use('classic')


# In[15]:


import seaborn as sns
sns.scatterplot(data=finalDf,s=100)  


# In[16]:


p1 = sns.scatterplot(data=finalDf,s = 100)  
for line in range(0,finalDf.shape[0]):
     p1.text(finalDf.pc1[line], finalDf.pc2[line], finalDf.pc3[line], finalDf.Type[line], horizontalalignment='left', size='large')
        


# ## CLUSTERING

# In[17]:


from sklearn.preprocessing import MinMaxScaler
trans = MinMaxScaler()
data=finalDf
data 


# In[18]:


from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 
#p = np.array(df_norm) # converting into numpy array format 
z = linkage(finalDf, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
    #leaf_rotation=0.,  # rotates the x axis labels
    #leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[19]:


from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=4, linkage='complete',affinity = "euclidean").fit(data) 

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
wine['clust']=cluster_labels # creating a  new column and assigning it to new column 
wine


# In[20]:


wine.iloc[:,1:].groupby(wine.clust).mean()


# ##BY USING KMEANS

# In[21]:


import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
import numpy as np


# In[22]:


X = np.random.uniform(0,1,1000)
Y = np.random.uniform(0,1,1000)
X


# In[23]:


df_xy =pd.DataFrame(columns=["X","Y"])
df_xy
df_xy.X = X
df_xy.Y = Y
df_xy
df_xy.plot(x="X",y = "Y",kind="scatter")
model1 = KMeans(n_clusters=5).fit(df_xy)


# In[24]:


df_xy.plot(x="X",y = "Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm_r)


# In[25]:


wine1=finalDf
wine1


# In[26]:


from sklearn.cluster import KMeans
fig = plt.figure(figsize=(10, 8))
WCSS = []
for i in range(1, 11):
    clf = KMeans(n_clusters=i)
    clf.fit(wine1)
    WCSS.append(clf.inertia_) # inertia is another name for WCSS
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()


# In[27]:


clf = KMeans(n_clusters=3)
y_kmeans = clf.fit_predict(wine)


# In[28]:


y_kmeans
#clf.cluster_centers_
clf.labels_


# In[33]:


md=pd.Series(y_kmeans)  # converting numpy array into pandas series object 
wine['clust']=md # creating a  new column and assigning it to new column 
wine


# In[34]:


wine.iloc[:,1:14].groupby(wine.clust).mean()


# In[36]:


wine.plot(x="Alcohol",y ="Ash",c=clf.labels_,kind="scatter",s=50 ,cmap=plt.cm.coolwarm) 
plt.title('Clusters using KMeans')


# In[37]:


## on pca score no of clusters in hclustering =4,and in kmeans no of clusters =3
## no of clusters are not same for both by using pca score


# In[ ]:




