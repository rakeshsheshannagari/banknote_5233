#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


# In[2]:


df=pd.read_csv('BankNote_Authentication.csv')


# In[6]:


X=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[7]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[8]:


classifier=RandomForestClassifier()


# In[9]:


classifier.fit(X_train,y_train)


# In[10]:


y_pred=classifier.predict(X_test)


# In[11]:


from sklearn.metrics import accuracy_score


# In[12]:


score=accuracy_score(y_test,y_pred)


# In[13]:


score


# In[14]:


pickle_out = open("BankNote.pickle","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()


# In[ ]:




