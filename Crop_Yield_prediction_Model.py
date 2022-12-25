#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing all necessary modules
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


df = pd.read_csv("yield_df.csv")
# df1=pd.read_csv("yield-changer.csv")   #to change the area,item to their codes
data = df.copy()  # making a copy of dataset


# In[4]:


data.isnull().sum()  # checking for count of missing values


# In[5]:


data.columns


# In[6]:


cr = LabelEncoder()
se = LabelEncoder()
ye = LabelEncoder()
data['Area'] = se.fit_transform(data['Area'])
data['Item'] = cr.fit_transform(data['Item'])
data['Year'] = ye.fit_transform(data['Year'])


# In[7]:


data.drop(columns=['Unnamed: 0'], inplace=True)


# In[8]:


final_data = data


# In[9]:


# Training and Testing data
X = final_data.drop('hg/ha_yield', axis=1)
y = final_data['hg/ha_yield']


# In[10]:


# Splitting the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)


# In[11]:


lin_reg = LinearRegression()
#lin_reg


# In[50]:


lin_reg.fit(X_train, y_train)


# In[51]:


y_pred = lin_reg.predict(X_train)


# In[13]:


lin_mse = mean_squared_error(y_train, y_pred)
lin_rmse = np.sqrt(lin_mse)
#lin_rmse


# In[54]:


scores = cross_val_score(lin_reg,
                         X_train,
                         y_train,
                         scoring="neg_mean_squared_error",
                         cv=10)
lin_reg_rmse_scores = np.sqrt(-scores)


# In[55]:


#lin_reg_rmse_scores


# In[15]:


tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)


# In[59]:


tree_pred = tree_reg.predict(X_train)


# In[60]:


tree_mse = mean_squared_error(y_train, tree_pred)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

# In[61]:


# Cross Validation and Hyperparameter tuning


# In[62]:


scores = cross_val_score(tree_reg,
                         X_train,
                         y_train,
                         scoring="neg_mean_squared_error",
                         cv=10)
tree_reg_rmse_scores = np.sqrt(-scores)


# In[63]:


#tree_reg_rmse_scores


# In[17]:


forest_reg = RandomForestRegressor()


# In[67]:

forest_reg.fit(X_train, y_train)


# In[68]:


forest_pred = forest_reg.predict(X_train)


# In[69]:


forest_mse = mean_squared_error(y_train, forest_pred)
forest_rmse = np.sqrt(forest_mse)
#forest_rmse


# In[70]:


# Cross Validation and Hyperparameter tuning


# In[71]:


forest_reg_cv_scores = cross_val_score(forest_reg,
                                       X_train,
                                       y_train,
                                       scoring="neg_mean_squared_error",
                                       cv=10)
forest_reg_rmse_scores = np.sqrt(-forest_reg_cv_scores)


# In[72]:


#forest_reg_rmse_scores


# In[73]:


#forest_reg_rmse_scores.mean()


# In[18]:


param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features':[2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators':[3, 10], 'max_features':[2, 3, 4]}]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg,
                           param_grid,
                           scoring="neg_mean_squared_error",
                           return_train_score=True,
                           cv=10)
grid_search.fit(X_train, y_train)


# In[78]:


#grid_search.best_params_


# In[79]:


cv_scores = grid_search.cv_results_
# printing all the parameters along with their scores
#for mean_score, params in zip(cv_scores['mean_test_score'], cv_scores['params']):
    #print(np.sqrt(-mean_score), params)


# In[80]:


# feature importances
feature_importances = grid_search.best_estimator_.feature_importances_
#feature_importances


# In[19]:


final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[20]:


import pickle
#saving the model
with open("model.bin",'wb')as f_out:
    pickle.dump(final_model,f_out)
    f_out.close()


# In[21]:


# Create a function to cover the entire flow
def predict_yield(config, model):
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    y_pred = model.predict(df)
    return y_pred


# In[ ]:




