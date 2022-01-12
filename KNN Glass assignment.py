#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np


# In[37]:


glass= pd.read_csv("F:/Dataset/glass.csv")


# In[38]:


glass


# In[39]:


glass.info()


# In[40]:


array=glass.values
X= array[:,0:9]


# In[41]:


X


# In[42]:


Y= array[:,9]


# In[43]:


Y


# In[44]:


from sklearn.model_selection import KFold


# In[45]:


KFold=KFold(n_splits=10)


# In[46]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# In[47]:


model=KNeighborsClassifier(n_neighbors=18)


# In[48]:


result= cross_val_score(model,X,Y,cv=KFold)


# In[49]:


print(result.mean())


# In[50]:


from sklearn.model_selection import GridSearchCV


# In[51]:


n_neighbors1=np.array(range(1,80))


# In[52]:


param_grid=dict(n_neighbors=n_neighbors)


# In[53]:


model=KNeighborsClassifier()


# In[54]:


grid = GridSearchCV(estimator=model, param_grid=param_grid)


# In[55]:


grid.fit(X, Y)


# In[56]:


print(grid.best_score_)


# In[57]:


print(grid.best_params_)


# In[58]:


import matplotlib.pyplot as plt


# In[59]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[60]:


k_range=range(1,80)


# In[61]:


k_scores=[]


# In[64]:


for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores= cross_val_score(knn,X,Y,cv=5)
    k_scores.append(scores.mean())


# In[65]:


plt.plot(k_range,k_scores)
plt.show


# In[ ]:




