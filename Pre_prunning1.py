#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import seaborn as sns
df=sns.load_dataset('iris')
from sklearn.datasets import load_iris
iris=load_iris()


# In[3]:


df.head()


# In[4]:


X=df.iloc[:,:-1]
y=iris.target


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[7]:


from sklearn.tree import DecisionTreeClassifier


# In[11]:


treemodel=DecisionTreeClassifier(max_depth=2)


# In[ ]:





# In[12]:


treemodel.fit(X_train,y_train)


# In[13]:


from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treemodel,filled=True)


# In[14]:


y_pred=treemodel.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))

