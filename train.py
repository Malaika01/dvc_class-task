#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score , classification_report
from sklearn.linear_model import LogisticRegression


# In[3]:


df =pd.read_csv('creditcard.csv')


# In[4]:


df.head()


# In[17]:


x = df.drop("Class" , axis = 1)
y = df['Class']


# In[18]:


from imblearn.over_sampling import SMOTE

# setting up testing and training sets

sm = SMOTE(sampling_strategy='minority')
x_sm ,y_sm = sm.fit_resample(x, y)


# In[19]:


lr = LogisticRegression(C=10,max_iter=500)


# In[21]:


x_train, x_test, y_train, y_test = train_test_split(x_sm ,y_sm ,test_size=0.20, random_state=27)


# In[23]:


lr.fit(x_train, y_train)


# In[24]:


lr.score(x_train, y_train)


# In[ ]:



