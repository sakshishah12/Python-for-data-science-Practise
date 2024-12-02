#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../../Pierian_Data_Logo.png' /></a>
# ___

# # SF Salaries Exercise 
# 
# Welcome to a quick exercise for you to practice your pandas skills! We will be using the [SF Salaries Dataset](https://www.kaggle.com/kaggle/sf-salaries) from Kaggle! Just follow along and complete the tasks outlined in bold below. The tasks will get harder and harder as you go along.

# ** Import pandas as pd.**

# In[31]:


import pandas as pd


# ** Read Salaries.csv as a dataframe called sal.**

# In[32]:


sal=pd.read_csv("Salaries.csv",low_memory=False)


# ** Check the head of the DataFrame. **

# In[33]:


sal.head()


# ** Use the .info() method to find out how many entries there are.**

# In[34]:


sal.info()


# **What is the average BasePay ?**

# In[35]:


sal['TotalPay'].mean()


# ** What is the highest amount of OvertimePay in the dataset ? **

# In[36]:


sal['OvertimePay'].max()


# ** What is the job title of  JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll). **

# In[37]:


sal[sal["EmployeeName"]=="JOSEPH DRISCOLL"]["JobTitle"]


# ** How much does JOSEPH DRISCOLL make (including benefits)? **

# In[38]:


sal[sal["EmployeeName"]=="JOSEPH DRISCOLL"]["TotalPayBenefits"]


# ** What is the name of highest paid person (including benefits)?**

# In[40]:


sal[sal["TotalPayBenefits"]==sal["TotalPayBenefits"].max()]["EmployeeName"]


# ** What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?**

# In[41]:


sal[sal["TotalPayBenefits"]==sal["TotalPayBenefits"].min()]["EmployeeName"]


# ** What was the average (mean) BasePay of all employees per year? (2011-2014) ? **

# In[44]:


sal.groupby("Year").mean()


# ** How many unique job titles are there? **

# In[45]:


sal["JobTitle"].nunique()


# ** What are the top 5 most common jobs? **

# In[47]:


sal["JobTitle"].value_counts().head()


# ** How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?) **

# In[55]:


sum(sal[sal["Year"]==2013]["JobTitle"].value_counts()==1)


# In[58]:


def chief_title(x):
    if 'chief'in x.lower().split():
        return True
    else:
        return False


# ** How many people have the word Chief in their job title? (This is pretty tricky) **

# In[60]:


sum(sal["JobTitle"].apply(lambda x: chief_title(x)))


# In[21]:





# ** Bonus: Is there a correlation between length of the Job Title string and Salary? **

# In[63]:


sal["title_len"]=sal["JobTitle"].apply(len)


# In[66]:


sal[["title_len","TotalPayBenefits"]].corr()


# # Great Job!
