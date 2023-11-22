#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score


# In[30]:


df = pd.read_csv("C:/Users/abhis/OneDrive/Documents/Data science/10-29-2023/pro_2/50_Startups.csv")
df.head()


# In[31]:


##convert state into numerical form or continuos form 
df['State'] = df['State'].astype('category')
df['State'] = df['State'].cat.codes
df


# ## Exploratory Data analysis

# In[32]:


df.info()


# In[33]:


df.isnull().sum()


# In[34]:


df.describe()


# In[35]:


df.corr()


# In[36]:


len(df)


# In[57]:


sns.pairplot(df)


# In[38]:


##convert state into numerical form 
df['State'] = df['State'].astype('category')
df['State'] = df['State'].cat.codes
df


# In[39]:


X = df[['R&D Spend','Administration','Marketing Spend','State']]    ### Or you we can drop profit from df
X.head(2)


# In[40]:


X1 = df.drop(columns ='Profit')
X1


# In[41]:


y = df['Profit']   ##defined y -variable 
y


# ## Model selection

# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state = 0)



# In[ ]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg


# In[ ]:


X_train


# In[ ]:


y_train


# In[53]:


reg.fit(X_train,y_train)
reg.fit


# In[ ]:


c = reg.intercept_
c 


# In[56]:


m = reg.coef_
print( "slope -", m)


# In[46]:


###predicting the test set rules
y_pred = reg.predict(X_train)
y_pred


# In[45]:


from sklearn.metrics import r2_score


# In[49]:


##checking performance of regression model
print("Thetraining accuracy is -")
r2_score(y_train,y_pred)


# In[79]:


df1  = pd.DataFrame({'R^2': [.950009880362248]})
df1


# In[ ]:




