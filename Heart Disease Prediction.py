#!/usr/bin/env python
# coding: utf-8

# # Importing Dependencies

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# importing dataset
data=pd.read_csv('heart.csv')


# # Analysing dataset

# In[3]:


# checking first 5 rows of dataset
data.head()


# In[4]:


# checking null values
data.isnull().sum()


# In[5]:


# checking distribution of target variable
data['target'].value_counts()


# no.of 1's and 0's are nearly equal so no need to balance data

# # Normalization

# In[9]:


X = data.drop(columns=['target'])
Y = data['target']


# In[13]:


from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)


# # Spliting dataset into train and test set

# In[14]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=23,stratify=Y)


# # Importing Logistic Regression 

# In[17]:


from sklearn.linear_model import LogisticRegression as lr
model = lr()


# # Training the model

# In[26]:


model.fit(X_train,Y_train)


# In[27]:


train_predictions = model.predict(X_train)


# In[28]:


model.score(X_train,Y_train)


# # Testing the model

# In[29]:


test_predictions = model.predict(X_test)


# In[30]:


model.score(X_test,Y_test)


# # Building predictive model

# In[37]:


# input_array = [67,1,0,160,286,0,0,108,1,1.5,1,3,2]
# input_nparray = np.asarray(input_array)
# input_nparray = input_nparray.reshape(1,-1)


# In[38]:


# predict = model.predict(input_nparray)
# print(predict)

# if(predict==1):
#     print('You have a heart disease just die')
# else:
#     print('You dont have a heart disease but you are a douchebag just die')


import pickle

pickle.dump(model,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))


