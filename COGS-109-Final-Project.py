#!/usr/bin/env python
# coding: utf-8

# In[130]:


# Import libraries
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np  
import statistics
import statsmodels.formula.api as smf
import statsmodels.formula.api as sm
import scipy.stats as sp
import seaborn as sns
sns.set_style('ticks', rc={'axes.grid':True})
sns.set_context('talk')
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from scipy import linalg
import math


# In[131]:


df = pd.read_csv('crimes.csv', sep=',')
df.head()


# In[132]:


df_new = pd.DataFrame(df, columns = ['Date', 'Year', 'Primary Type', 'Community Area'])
df_new.head()


# In[133]:


df_new = df_new.sort_values(by='Date')
df_new.head()


# In[134]:


df_new = df_new.dropna()
df_new.head()


# In[135]:


index = df_new[df_new['Year'] < 2015].index
df_new.drop(index , inplace=True)
df_new.head()


# In[166]:


df_2015 = df_new[df_new['Year'] == 2015]
df_2015 = df_2015.sort_values(by='Community Area')
df_2015_0 = df_2015.loc[df['Community Area'] == 0.0]
temp = df_2015_0['Primary Type'].value_counts()
a = []
foo = df_new['Primary Type'].value_counts()

for index, val in foo.iteritems():
    a.append(index)
b = []

for i in range (0,33):
    bool = 1
    for index, val in temp.iteritems():
        if a[i] == index:
            b.append(val)
            bool = 1
            break
        else:
            bool = 0
    if bool == 0:
        b.append(0)
len(b)


# In[186]:


df_2015 = df_new[df_new['Year'] == 2015]
df_2015 = df_2015.sort_values(by='Community Area')

a = []
foo = df_new['Primary Type'].value_counts()
for index, val in foo.iteritems():
    a.append(index)

data2015 = pd.DataFrame()
for k in range(0,77):
    df_2015_i = df_2015.loc[df['Community Area'] == k]
    temp = df_2015_i['Primary Type'].value_counts()
    b = []
    for i in range (0,33):
        bool = 1
        for index, val in temp.iteritems():
            if a[i] == index:
                b.append(val)
                bool = 1
                break
            else:
                bool = 0
        if bool == 0:
            b.append(0)
    newb = pd.DataFrame(b)
    newb = newb.T
    data2015 = data2015.append(newb)
data2015.head()


# In[189]:


df_2016 = df_new[df_new['Year'] == 2016]
df_2016 = df_2016.sort_values(by='Community Area')

a = []
foo = df_new['Primary Type'].value_counts()
for index, val in foo.iteritems():
    a.append(index)

data2016 = pd.DataFrame()
for k in range(0,77):
    df_2015_i = df_2016.loc[df['Community Area'] == k]
    temp = df_2015_i['Primary Type'].value_counts()
    b = []
    for i in range (0,33):
        bool = 1
        for index, val in temp.iteritems():
            if a[i] == index:
                b.append(val)
                bool = 1
                break
            else:
                bool = 0
        if bool == 0:
            b.append(0)
    newb = pd.DataFrame(b)
    newb = newb.T
    data2016 = data2016.append(newb)
data2016.head()


# In[190]:


df_2017 = df_new[df_new['Year'] == 2017]
df_2017 = df_2017.sort_values(by='Community Area')

a = []
foo = df_new['Primary Type'].value_counts()
for index, val in foo.iteritems():
    a.append(index)

data2017 = pd.DataFrame()
for k in range(0,77):
    df_2015_i = df_2017.loc[df['Community Area'] == k]
    temp = df_2015_i['Primary Type'].value_counts()
    b = []
    for i in range (0,33):
        bool = 1
        for index, val in temp.iteritems():
            if a[i] == index:
                b.append(val)
                bool = 1
                break
            else:
                bool = 0
        if bool == 0:
            b.append(0)
    newb = pd.DataFrame(b)
    newb = newb.T
    data2017 = data2017.append(newb)
data2017.head()


# In[191]:


df_2018 = df_new[df_new['Year'] == 2018]
df_2018 = df_2018.sort_values(by='Community Area')

a = []
foo = df_new['Primary Type'].value_counts()
for index, val in foo.iteritems():
    a.append(index)

data2018 = pd.DataFrame()
for k in range(0,77):
    df_2015_i = df_2018.loc[df['Community Area'] == k]
    temp = df_2015_i['Primary Type'].value_counts()
    b = []
    for i in range (0,33):
        bool = 1
        for index, val in temp.iteritems():
            if a[i] == index:
                b.append(val)
                bool = 1
                break
            else:
                bool = 0
        if bool == 0:
            b.append(0)
    newb = pd.DataFrame(b)
    newb = newb.T
    data2018 = data2018.append(newb)
data2018.head()


# In[193]:


df_2019 = df_new[df_new['Year'] == 2019]
df_2019 = df_2019.sort_values(by='Community Area')

a = []
foo = df_new['Primary Type'].value_counts()
for index, val in foo.iteritems():
    a.append(index)

data2019 = pd.DataFrame()
for k in range(0,77):
    df_2015_i = df_2019.loc[df['Community Area'] == k]
    temp = df_2015_i['Primary Type'].value_counts()
    b = []
    for i in range (0,33):
        bool = 1
        for index, val in temp.iteritems():
            if a[i] == index:
                b.append(val)
                bool = 1
                break
            else:
                bool = 0
        if bool == 0:
            b.append(0)
    newb = pd.DataFrame(b)
    newb = newb.T
    data2019 = data2019.append(newb)
data2019


# In[196]:


FinalData = pd.DataFrame()
FinalData = FinalData.append(data2015)
FinalData = FinalData.append(data2017)
FinalData = FinalData.append(data2018)
FinalData = FinalData.append(data2019)
FinalData


# In[204]:


#FinalData.to_excel('output1.xlsx')


# In[ ]:




