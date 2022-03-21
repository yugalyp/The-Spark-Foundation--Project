#!/usr/bin/env python
# coding: utf-8

# # Name : Yugal Pachpande
# 
# ## Task 2 : Prediction using Unsupervised Machine Learning
# ## GRIP @ The Sparks Foundation
# 
# In this K-means clustering task I tried to predict the optimum number of clusters and represent it visually from the given ‘Iris’ dataset.
# 
# 
# ## Technical Stack  : Scikit Learn, Numpy Array, Scipy, Pandas, Matplotlib

# In[1]:


from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
import sklearn.metrics as sm
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import DBSCAN 
from sklearn.decomposition import PCA 


# ## Step 1 - Loading the dataset

# In[2]:


iris = datasets.load_iris()
print(iris.data)


# In[3]:


print(iris.target_names)


# In[4]:


print(iris.target)


# In[5]:


x = iris.data
y = iris.target


# ## Step 2 - Visualizing the input data and its Hierarchy

# In[6]:


#Plotting
fig = plt.figure(1, figsize=(7,5))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(x[:, 3], x[:, 0], x[:, 2], edgecolor="k", s=50)
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("Iris Clustering K Means=3", fontsize=14)
plt.show()

#Hierachy Clustering 
hier=linkage(x,"ward")
max_d=7.08
plt.figure(figsize=(15,8))
plt.title('Iris Hierarchical Clustering Dendrogram')
plt.xlabel('Species')
plt.ylabel('distance')
dendrogram(
    hier,
    truncate_mode='lastp',  
    p=50,                  
    leaf_rotation=90.,      
    leaf_font_size=8.,     
)
plt.axhline(y=max_d, c='k')
plt.show()


# ## Step 3 - Data Preprocessing

# In[7]:


x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])


# In[8]:


x.head()


# In[9]:


y.head()


# ## Step 4 - Model Training 

# In[11]:


iris_k_mean_model = KMeans(n_clusters=3)
iris_k_mean_model.fit(x)


# In[12]:


print(iris_k_mean_model.labels_)


# In[13]:


print(iris_k_mean_model.cluster_centers_)


# ## Step 5 - Visualizing the Model Cluster 

# In[14]:


plt.figure(figsize=(14,6))

colors = np.array(['red', 'green', 'blue'])

predictedY = np.choose(iris_k_mean_model.labels_, [1, 0, 2]).astype(np.int64)

plt.subplot(1, 2, 1)
plt.scatter(x['Petal Length'], x['Petal Width'], c=colors[y['Target']])
plt.title('Before classification')
plt.legend(handles=[red_patch, green_patch, blue_patch])

plt.subplot(1, 2, 2)
plt.scatter(x['Petal Length'], x['Petal Width'], c=colors[predictedY])
plt.title("Model's classification")
plt.legend(handles=[red_patch, green_patch, blue_patch])


# ## Step 6 - Calculating the Accuracy and Confusion Matrix

# In[15]:


sm.accuracy_score(predictedY, y['Target'])


# In[16]:


sm.confusion_matrix(predictedY, y['Target'])


# In a confusion matrix, the predicted class labels (0, 1, 2) are written along the top (column names). The true class labels (Iris-setosa, etc.) are written along the right side. Each cell in the matrix is a count of how many instances of a true class where classified as each of the predicted classes.
# &nbsp;
# 
# ## Conclusion
# ### I was  able to successfully carry-out prediction using Unsupervised Machine Learning task and was able to evaluate the model's clustering accuracy score.
# # Thank You
