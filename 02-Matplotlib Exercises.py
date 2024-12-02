#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Matplotlib Exercises 
# 
# Welcome to the exercises for reviewing matplotlib! Take your time with these, Matplotlib can be tricky to understand at first. These are relatively simple plots, but they can be hard if this is your first time with matplotlib, feel free to reference the solutions as you go along.
# 
# Also don't worry if you find the matplotlib syntax frustrating, we actually won't be using it that often throughout the course, we will switch to using seaborn and pandas built-in visualization capabilities. But, those are built-off of matplotlib, which is why it is still important to get exposure to it!
# 
# ** * NOTE: ALL THE COMMANDS FOR PLOTTING A FIGURE SHOULD ALL GO IN THE SAME CELL. SEPARATING THEM OUT INTO MULTIPLE CELLS MAY CAUSE NOTHING TO SHOW UP. * **
# 
# # Exercises
# 
# Follow the instructions to recreate the plots using this data:
# 
# ## Data

# In[13]:


import numpy as np
x = np.arange(0,100)
y = x*2
z = x**2


# ** Import matplotlib.pyplot as plt and set %matplotlib inline if you are using the jupyter notebook. What command do you use if you aren't using the jupyter notebook?**

# In[1]:


import matplotlib.pyplot as plt


# ## Exercise 1
# 
# ** Follow along with these steps: **
# * ** Create a figure object called fig using plt.figure() **
# * ** Use add_axes to add an axis to the figure canvas at [0,0,1,1]. Call this new axis ax. **
# * ** Plot (x,y) on that axes and set the labels and titles to match the plot below:**

# In[3]:


fig=plt.figure()
axes=fig.add_axes([0,0,1,1])
axes.plot(x,y)


# ## Exercise 2
# ** Create a figure object and put two axes on it, ax1 and ax2. Located at [0,0,1,1] and [0.2,0.5,.2,.2] respectively.**

# In[4]:


fig1=plt.figure()
axes1=fig1.add_axes([0,0,1,1])
axes2=fig1.add_axes([0.2,0.5,0.2,0.2])


# ** Now plot (x,y) on both axes. And call your figure object to show it.**

# In[7]:


axes1.plot(x,y)
axes2.plot(x,y)
fig1


# ## Exercise 3
# 
# ** Create the plot below by adding two axes to a figure object at [0,0,1,1] and [0.2,0.5,.4,.4]**

# In[15]:


fig2=plt.figure()
ax1=fig2.add_axes([0,0,1,1])
ax2=fig2.add_axes([0.2,0.5,0.4,0.4])


# ** Now use x,y, and z arrays to recreate the plot below. Notice the xlimits and y limits on the inserted plot:**

# In[17]:


# x=np.array([20,40,60,80,100])
# y=np.array([40,41,42,43,44])
# z=np.array([2000,4000,6000,8000,10000])
ax1.plot(x,z)
ax1.set_xlim(20,100)
ax2.plot(x,y)
fig2


# ## Exercise 4
# 
# ** Use plt.subplots(nrows=1, ncols=2) to create the plot below.**

# In[18]:


fig3,axes3=plt.subplots(nrows=1,ncols=2)


# ** Now plot (x,y) and (x,z) on the axes. Play around with the linewidth and style**

# In[20]:


axes3[0].plot(x,y,linestyle="--",color="blue",linewidth=3)
axes3[1].plot(x,z,linestyle="solid",color="red",linewidth=3)
fig3


# ** See if you can resize the plot by adding the figsize() argument in plt.subplots() are copying and pasting your previous code.**

# In[24]:


fig3,axes3=plt.subplots(nrows=1,ncols=2,figsize=(12,2))
axes3[0].plot(x,y,linestyle="--",color="blue",linewidth=3)
axes3[1].plot(x,z,linestyle="solid",color="red",linewidth=3)


# # Great Job!
