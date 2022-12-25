#!/usr/bin/env python
# coding: utf-8

# # Crop-Yield-Prediction

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


# Importing the dataset-Data Collection

# In[3]:


df = pd.read_csv("yield_df.csv")
# df1=pd.read_csv("yield-changer.csv")   #to change the area,item to their codes
data = df.copy()  # making a copy of dataset


# Dataset insights

# In[4]:


data


# In[5]:


# sample() returns random rows from dataframe
data.sample(10)


# #Problem Statement: The data contains hg/ha_yield variable which is continuous data and tells us about the crop yield in area
#
# Our aim here is to predict the hg/ha_yield given various parameters like Area,Item,average_rain_fall_mm_per_year,pesticides_tonnes and avg_temp
#
# The name of the crop is determined by several features like temperature, humidity, wind speed, rainfall, etc. and yield is determined by the area and production.

# In[6]:


data.info()


# In[7]:


data.isnull().sum()  # checking for count of missing values


# No null values are present ((i.e) No missing data is present)

# In[8]:


data.describe()


# In[9]:


data.columns


# In[10]:


data['Area'].value_counts()/len(data)


# In[11]:


data['Item'].value_counts()/len(data)


# Data Visualisation

# In[12]:


# sns.countplot(df['Item'])
# plt.figure()
# plt.show()


# In[13]:


# x = df["Area"]
# y = df['hg/ha_yield']
# plt.plot(x, y)
# plt.show()


# In[14]:


# data['hg/ha_yield'].plot.hist()
# plt.show()


# In[15]:


# sns.boxplot(x="Year", y="hg/ha_yield", data=data)
# plt.show()


# In[16]:


data.columns


# In[17]:


cr = LabelEncoder()
se = LabelEncoder()
ye = LabelEncoder()
data['Area'] = se.fit_transform(data['Area'])
data['Item'] = cr.fit_transform(data['Item'])
data['Year'] = ye.fit_transform(data['Year'])


# In[18]:


data.head(10)


# In[19]:


data.sample(10)


# In[20]:


# We will plot pairplots to get intuition of potential correlations
# sns.pairplot(data[['Area', 'Item', 'Year', 'hg/ha_yield',
#                    'average_rain_fall_mm_per_year', "pesticides_tonnes", "avg_temp"]], diag_kind='kde')
# plt.show()  # pairwise distribution for every column
# The pair plot gives you a brief overview of how each variable behaves with respect to every other variable.


# In[21]:


data.columns


# dropping unnecessary columns

# In[22]:


data.drop(columns=['Unnamed: 0'], inplace=True)


# In[23]:


data.columns  # column is removed


# In[24]:


# df1.sample(10)


# In[25]:


# df1["Item"].value_counts()


# In[26]:


# df1["Area"].value_counts()


# In[27]:


# df1.columns


# In[28]:


# df1=df1.drop(columns=['Domain Code', 'Domain','Element Code', 'Element','Year Code', 'Year', 'Unit', 'Value'])


# In[29]:


# df1


# In[30]:


# df_cd=data.merge(df1,on=['Area',"Item"])


# In[31]:


# df_cd.isnull().sum()


# In[32]:


# df_cd['Area'].value_counts()


# In[33]:


# df_cd.columns


# In[34]:


# df_cd.drop(columns=['Area', 'Item', 'Year'],inplace=True)


# In[35]:


final_data = data
final_data


# Train-test Split

# In[36]:


# Training and Testing data
X = final_data.drop('hg/ha_yield', axis=1)
y = final_data['hg/ha_yield']


# In[37]:


# Splitting the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)


# In[38]:


X_train.head()


# In[39]:


y_train.head()


# In[40]:


X_test.head()


# In[41]:


y_test.head()


# In[42]:


print(X_train.shape, X_test.shape)


# In[43]:


X_train["Item"].value_counts()/len(X_train)


# In[44]:


# All the categories are present in X_test as well
X_test["Item"].value_counts()/len(X_test)


# correlation matrix

# In[45]:


# Checking the correlation matrix w.r.t yg_ya field
# corr_matrix = final_data.corr()


# In[46]:


# corr_ = corr_matrix['hg/ha_yield'].sort_values(ascending=False)
# corr_


# In[47]:


# X_train.corr()


# In[48]:


# pd.DataFrame(corr_).style.background_gradient(cmap='coolwarm')


# # Selecting and Training Models

# Linear Regression

# In[49]:

print("linear regression")
lin_reg = LinearRegression()
lin_reg


# In[50]:


lin_reg.fit(X_train, y_train)


# In[51]:


y_pred = lin_reg.predict(X_train)


# In[52]:


# Mean Squared Error


# In[53]:


lin_mse = mean_squared_error(y_train, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[54]:


scores = cross_val_score(lin_reg,
                         X_train,
                         y_train,
                         scoring="neg_mean_squared_error",
                         cv=10)
lin_reg_rmse_scores = np.sqrt(-scores)


# In[55]:


lin_reg_rmse_scores


# In[56]:


lin_reg_rmse_scores.mean()


# In[57]:


# Decision Tree


# In[58]:

print("Decision Tree")
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


tree_reg_rmse_scores


# In[64]:


tree_reg_rmse_scores.mean()


# In[65]:


# Random Forest


# In[66]:

print("random forest regressor")
forest_reg = RandomForestRegressor()


# In[67]:


forest_reg.fit(X_train, y_train)


# In[68]:


forest_pred = forest_reg.predict(X_train)


# In[69]:


forest_mse = mean_squared_error(y_train, forest_pred)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


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


forest_reg_rmse_scores


# In[73]:


forest_reg_rmse_scores.mean()


# In[74]:


# Support Vector Machine Regressor


# In[84]:

# print("SVM")
# svm_reg = SVR(kernel="linear")


# In[ ]:


# svm_reg.fit(X_train,y_train)


# In[ ]:


# svm_reg_cv_scores = cross_val_score(svm_reg,
#                          X_train,
#                          y_train,
#                          scoring ="neg_mean_squared_error",
#                          cv=10)
# svm_reg_rmse_scores = np.sqrt(-svm_reg_cv_scores)


# In[ ]:


# svm_reg_rmse_scores


# In[ ]:


# svm_reg_rmse_scores .mean()


# # Fine-Tuning Hyperparameters
#
# Hyperparameter Tuning using GridSearchCV
#
# After testing all the models, you’ll find that RandomForestRegressor has performed the best but it still needs to be fine-tuned.
#
# A model is like a radio station with a lot of knobs to handle and tune. Now, you can either tune all these knobs manually or provide a range of values/combinations that you want to test.
#
# We use GridSearchCV to find out the best combination of hyperparameters for the RandomForest model
#

# In[76]:


# In[77]:


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


grid_search.best_params_


# In[79]:


cv_scores = grid_search.cv_results_
# printing all the parameters along with their scores
for mean_score, params in zip(cv_scores['mean_test_score'], cv_scores['params']):
    print(np.sqrt(-mean_score), params)


# In[80]:


# feature importances
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[81]:


# Evaluating the Entire System
print("Final_model")
final_model = grid_search.best_estimator_


# In[83]:

print("final predictions")
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[94]:


# Create a function to cover the entire flow
def predict_yield(config, model):
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    y_pred = model.predict(df)
    return y_pred


# In[95]:


data.columns


# In[96]:


# checking the above flow on a random sample
# param_config = {
#     "Area": [0],
#     "Item": [1],
#     "Year": [0],
#     "average_rain_fall_mm_per_year": [1485.0],
#     'pesticides_tonnes': [121.00],
#     'avg_temp': [16.37]
# }


# In[97]:

print("Last ")
# predict_yield(param_config, final_model)


# In[ ]:
