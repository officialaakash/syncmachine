#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pickle
import requests
import json


# In[36]:


df=pd.read_csv("SynchronousMachine.csv")


# In[37]:


df.head()


# In[38]:


new_col_names={'I_y': 'Load_Current','PF': 'Powerfactor','e_PF': 'Power_factor_error','d_if': 'Change_of_excitation_current','I_f': 'Excitation_ current'}


# In[39]:


df.columns=df.columns.map(new_col_names)


# In[40]:


df.head()


# In[41]:


df.info()


# In[42]:


df.corr()['Excitation_ current']


# In[43]:


sns.pairplot(df)


# In[44]:


def outlier(cols):
    
    for col in cols:
        print("Details of outliers for column: ",col)
        print('\n')
        
        q75,q25=np.percentile(df[col],[75,25])
    
        print("25th Percentile is: ",q25)
        print("75th Percentile is: ",q75)

        iqr=np.round(q75-q25,3)
        print("Inter Quartile range is: ",iqr)

        upperlim=np.round(q75+(1.5*iqr),3)
        print("Upper limit is: ", upperlim)

        lowerlim=np.round(q25-(1.5*iqr),3)
        print("Lower limit is: ", lowerlim)
        
        
        print('\n')
        print("Rows above Upper Limit are: ",len(df[df[col]>upperlim]))
        print("Rows below Lower Limit are: ",len(df[df[col]<lowerlim]))
        
        print('\n')
        print('-------------')
        print('\n')


# In[45]:


cols=df.columns


# In[46]:


outlier(cols)


# In[47]:


#LINEAR REGRESSION


# In[48]:


X=df.drop('Excitation_ current',axis=1)


# In[49]:


y=df['Excitation_ current']


# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[52]:


from sklearn.preprocessing import StandardScaler


# In[53]:


scaler= StandardScaler()


# In[54]:


scaler.fit(X_train)


# In[55]:


X_train=scaler.transform(X_train)


# In[56]:


X_test=scaler.transform(X_test)


# In[57]:


from sklearn.linear_model import LinearRegression


# In[58]:


model=LinearRegression()


# In[59]:


model.fit(X_train,y_train)


# In[60]:


predicted_y=model.predict(X_test)


# In[61]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[62]:


RMSE=np.sqrt(mean_squared_error(y_test,predicted_y))
MAE=mean_absolute_error(y_test,predicted_y)


# In[63]:


RMSE


# In[64]:


MAE


# In[65]:


model.score(X_test,y_test)


# In[66]:


print("Test Accuracy of Linear Regression model is:",round(100*model.score(X_test, y_test),2))


# In[87]:


pickle.dump(model, open('model.pkl','wb'))


# In[88]:


print(model.predict([[3.0, 0.66, 0.34,0.383]]))


# In[74]:


df.head()
