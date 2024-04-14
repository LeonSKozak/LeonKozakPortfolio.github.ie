#!/usr/bin/env python
# coding: utf-8

# ## LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score #evaluation of model performance
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer #Processing Pipeline
from sklearn.pipeline import Pipeline #Processing Pipeline
from sklearn.ensemble import RandomForestRegressor # Random Forrests
from sklearn.cluster import KMeans #K Means
from sklearn.decomposition import TruncatedSVD


# ## Data Set

# In[2]:


car_data = pd.read_csv('/Users/leonkozak/Documents/car_prices.csv')
car_data.head()


# In[3]:


car_data.describe().round()


# ## DATA CLEANING

# In[4]:


#dropping of missing values 
car_data.dropna(inplace=True)


# In[5]:


car_data.isna().sum()


# ## SUPERVISED ALGORITHM

# In[6]:


y = car_data['sellingprice']

feature_columns = ['year', 'make', 'model', 'odometer', 'condition']
X = car_data[feature_columns] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['make', 'model']
numerical_features = ['year', 'odometer', 'condition']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())])

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(y_pred[:5])  
print(f'Linear Regression MSE: {mean_squared_error(y_test, y_pred)}')
print(f'Linear Regression R² score: {r2_score(y_test, y_pred)}')


# #### The model's predicts with an average squared error of 41.1 million, which indicates large deviations from real selling prices.
# #### The R² score of 0.547 shows, that the model can explain 54.7% of the variance in car selling prices from the selected features.

# In[ ]:


#update the model pipeline to use a random forrest regressor 
y = car_data['sellingprice']
X = car_data.drop(['sellingprice', 'vin', 'saledate'], axis=1)

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
print(y_pred[:5]) 
print(f'Random Forest Regression MSE: {mean_squared_error(y_test, y_pred)}')
print(f'Random Forest Regression R² score: {r2_score(y_test, y_pred)}')


# #### The Random Forest model can significantly improve the prediction accuracy  (MSE of ~2.24 million and it explains ~97.53% of the variance in car selling prices (R² score) --> much better performance over all.
# #### The predicted selling prices for the first five cars in the test set range from approximately  4.573 USD to 23.044 USD, which indicates varied price predictions across the dataset.

# ## UNSUPERVISED ALGORTHM 

# In[7]:


y = car_data['sellingprice']
X = car_data.drop('sellingprice', axis=1)

# Preprocessing
categorical_features = ['make', 'model']
numerical_features = ['year', 'odometer', 'condition']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Elbow Method to determin number of clusters
# Iterating thorugh a range of 1 to 10 (k) clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('cluster', kmeans)])
    pipeline.fit(X)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

pipeline.fit(X) #Pipelines helps to simplify steps of data preprocessing to get final model


# In[8]:


# optimal_k determined based on Elbow method
optimal_k = 4  
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('cluster', kmeans)])


# In[9]:


X = car_data.drop('sellingprice', axis=1)

categorical_features = ['make', 'model']
numerical_features = ['year', 'odometer', 'condition']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('cluster', KMeans(n_clusters=3, random_state=42))  # Example: n_clusters set to 3
])

pipeline.fit(X)

svd = TruncatedSVD(n_components=2, random_state=42)
X_reduced = svd.fit_transform(pipeline.named_steps['preprocessor'].transform(X))

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=pipeline.named_steps['cluster'].labels_, cmap='viridis')
plt.title('2D Visualization of Car Data Clusters')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Cluster Label')
plt.show()


# ### Interactive plot

# In[10]:


import plotly.express as px

df = pd.DataFrame(X_reduced, columns=['Component 1', 'Component 2'])
df['Cluster'] = pipeline.named_steps['cluster'].labels_

fig = px.scatter(df, x='Component 1', y='Component 2', color='Cluster', title='Interactive Visualization of Data Clusters')
fig.show()


# In[14]:


import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('cluster', KMeans(n_clusters=5, random_state=42))  # showing 5 clusters/centroids])

pipeline.fit(X)

svd = TruncatedSVD(n_components=2, random_state=42)
X_reduced = svd.fit_transform(pipeline.named_steps['preprocessor'].transform(X))

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=pipeline.named_steps['cluster'].labels_, cmap='viridis')

centers_reduced = svd.transform(pipeline.named_steps['cluster'].cluster_centers_)

# Including Cluster Centers (Centroids)
plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], c='red', s=200, alpha=0.5, marker='X')
plt.title('2D Visualization of Car Data Clusters including Centroids')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Cluster Label')
plt.show()


# #### Interestingly, one of the centroids is located in an area with very little clusters (purple). Upon further research, I found out that this is common phenomenon that results from the dimensionality reduction process (from many features to just 2). K-Means operates in the "original" 3-Dimensional space, while the visualization is limited to 2 Dimensions.
# #### Therefore, I decided to plot it again in a 3-Dimensional way.

# In[24]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_reduced_3d[:, 0], X_reduced_3d[:, 1], X_reduced_3d[:, 2], c=labels, cmap='viridis', edgecolor='k', s=60, alpha=0.75)

ax.set_title('3D Visualization of Car Data Clusters')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Cluster Label')

plt.show()

