#!/usr/bin/env python
# coding: utf-8

# # Name : Yugal Pachpande
# 
# ## Task 3 : Prediction using Decision Tree Algorithm
# ## GRIP @ The Sparks Foundation
# 
# Decision Trees are versatile Machine Learning algorithms that can perform
# both classification and regression tasks, and even multioutput tasks.For the given ‘Iris’ dataset, I created the Decision Tree classifier and visualized it
# graphically. The purpose of this task is if we feed any new data to this classifier, it would be able to
# predict the right class accordingly.
# &nbsp;
# 
# 
# ## Technical Stack  : Sikit Learn, Numpy Array, Seaborn, Pandas, Matplotlib, Pydot
# <img src = img\1.png >

# In[1]:


# Importing the required Libraries

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import pydot
from IPython.display import Image


# ## Step 1 - Loading the Dataset

# In[2]:


# Loading Dataset
iris = load_iris()
X=iris.data[:,:] 
y=iris.target


# ## Step 2 - Exploratory Data Analysis

# In[3]:


#Input data 

data=pd.DataFrame(iris['data'],columns=["Petal length","Petal Width","Sepal Length","Sepal Width"])
data['Species']=iris['target']
data['Species']=data['Species'].apply(lambda x: iris['target_names'][x])

data.head()


# In[4]:


data.shape


# In[5]:


data.describe()


# ## Step 3 - Data Visualization comparing various features

# In[6]:


# Input data Visualization
sns.pairplot(data)


# In[7]:


# Scatter plot of data based on Sepal Length and Width features
sns.FacetGrid(data,hue='Species').map(plt.scatter,'Sepal Length','Sepal Width').add_legend()
plt.show()

# Scatter plot of data based on Petal Length and Width features
sns.FacetGrid(data,hue='Species').map(plt.scatter,'Petal length','Petal Width').add_legend()
plt.show()


# ## Step 4 - Decision Tree Model Training

# In[8]:


# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1) 
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train,y_train)
print("Training Complete.")
y_pred = tree_classifier.predict(X_test)


# ## Step 5 - Comparing the actual and predicted flower classification

# In[9]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 
df 


# ## Step 6 - Visualizing the Trained Model

# In[10]:


#Visualizing the trained Decision Tree Classifier taking all 4 features in consideration

export_graphviz(
        tree_classifier,
        out_file="img\desision_tree.dot",
        feature_names=iris.feature_names[:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
)

(graph,) = pydot.graph_from_dot_file('img\desision_tree.dot')
graph.write_png('img\desision_tree.png')

Image(filename='img\desision_tree.png') 


# ## Step 7 - Predicting the class output for some random values of petal and sepal length and width

# In[11]:


print("Class Names = ",iris.target_names)

# Estimating class probabilities
print()
print("Estimating Class Probabilities for flower whose petals length width are 4.7cm and 3.2cm and sepal length and width are 1.3cm and 0.2cm. ")
print()
print('Output = ',tree_classifier.predict([[4.7, 3.2, 1.3, 0.2]]))
print()
print("Our model predicts the class as 0, that is, setosa.")


# ## Step 8 - Calculating the Model accuracy

# In[12]:


# Model Accuracy
print("Accuracy:",sm.accuracy_score(y_test, y_pred))


# The accuracy of this model is 1 or 100% since I have taken all the 4 features of the iris dataset for creating the decision tree model.
# 
# ## Conclusion
# ### I was  able to successfully carry-out prediction using Prediction using Decision Tree Algorithm and was able to evaluate the model's accuracy score.
# # Thank You
