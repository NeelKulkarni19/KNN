#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


zoo = pd.read_csv('F:/Dataset/Zoo.csv')


# In[3]:


zoo


# In[9]:


zoo.info()


# In[13]:


zoo.drop('animal name',axis=1,inplace=True)


# In[14]:


array=zoo.values
X= array[:,1:18]


# In[15]:


X


# In[34]:


Y= array[:,16]


# In[27]:


Y


# In[28]:


from sklearn.model_selection import KFold


# In[29]:


KFold=KFold(n_splits=10)


# In[30]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# In[31]:


model=KNeighborsClassifier(n_neighbors=18)


# In[35]:


result= cross_val_score(model,X,Y,cv=KFold)


# In[36]:


print(result.mean())


# In[37]:


from sklearn.model_selection import GridSearchCV


# In[40]:


n_neighbors=np.array(range(1,80))


# In[41]:


paramgrid= dict(n_neighbors=n_neighbors)


# In[42]:


model = KNeighborsClassifier()


# In[45]:


grid= GridSearchCV(estimator=model,param_grid=paramgrid)


# In[46]:


grid.fit(X,Y)


# In[47]:


print(grid.best_score_)


# In[48]:


print(grid.best_params_)


# In[49]:


import matplotlib.pyplot as plt


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[51]:


k_range=range(1,80)


# In[52]:


k_scores=[]


# In[53]:


for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores= cross_val_score(knn,X,Y,cv=5)
    k_scores.append(scores.mean())


# In[54]:


plt.plot(k_range,k_scores)
plt.show


# In[ ]:




