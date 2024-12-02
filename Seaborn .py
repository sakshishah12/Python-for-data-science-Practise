#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns


# In[2]:


tips=sns.load_dataset('tips')


# In[3]:


tips.head()


# In[4]:


#DISTRIBUTED PLOTS
sns.distplot(tips["total_bill"])


# In[5]:


sns.jointplot(x='total_bill',y='tip',data=tips,kind='kde')


# In[6]:


sns.pairplot(tips,hue="sex") #use hue argument for categorical data

sns.rugplot(tips["total_bill"])
# In[7]:


sns.rugplot(tips["total_bill"])


# In[8]:


#CATEGORICAL PLOTS
import numpy as np
sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std)  #one axis is categorical data and other is numerical data and the plot is basically an estimator functioni.e an aggregate function


# In[9]:


sns.countplot(x="sex",data=tips)    #works as pandas count function


# In[10]:


sns.boxplot(x="day",y="total_bill",data=tips,hue="smoker")


# In[11]:


sns.violinplot(x="day",y="total_bill",data=tips,hue="sex",split=True)


# In[12]:


sns.stripplot(x="day",y="total_bill",data=tips,hue="sex")


# In[13]:


#MATRIX PLOTS
flights=sns.load_dataset("flights")
tc=tips.corr()
sns.heatmap(tc,annot=True,cmap="coolwarm")


# In[14]:


flights.head()


# In[15]:


fp=flights.pivot_table(index="month",columns="year",values="passengers")
sns.heatmap(fp,linecolor="white",linewidth="0.5")


# In[16]:


sns.clustermap(fp)


# In[17]:


import seaborn as sns
iris=sns.load_dataset("iris")
iris.head()


# In[18]:


sns.pairplot(iris)


# In[23]:


import matplotlib.pyplot as plt
g=sns.PairGrid(iris)
g.map_diag(sns.histplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)


# In[28]:


tips=sns.load_dataset("tips")
g2=sns.FacetGrid(data=tips,row="smoker",col="time")
g2.map(plt.scatter,'total_bill','tip')


# In[33]:


sns.lmplot(data=tips,x='total_bill',y='tip',hue='sex',markers=['o','v'],scatter_kws={'s':100})


# In[37]:


sns.lmplot(data=tips,x='total_bill',y='tip',row='time',col='sex')


# In[43]:


sns.lmplot(data=tips,x='total_bill',y='tip',col='day',hue='sex',aspect=0.5,height=8)


# In[50]:


plt.figure(figsize=(12,3))
sns.set_style('whitegrid')
sns.countplot(x='sex',data=tips)

