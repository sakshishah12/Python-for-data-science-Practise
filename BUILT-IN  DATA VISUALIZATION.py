#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[10]:


import seaborn as sns


# In[13]:


df1=pd.read_csv("df1",index_col=0)
df2=pd.read_csv("df2")
df3=pd.read_csv("df3")


# In[14]:


df1.head()


# In[15]:


df2.head()


# In[7]:


df3.head()


# In[16]:


df1['A'].hist(bins=30)


# In[18]:


df2.plot.area(alpha=0.4)


# In[ ]:





# In[ ]:




