#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.pyplot import figure
import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr

df = pd.read_csv('/Users/leonkozak/Documents/movies.csv')
df.head()


# In[ ]:





# In[6]:


# Checking for missing values
df.isna().any()


# In[ ]:


# There are missing values in many columns. Upon checking, I decided to drop these specific rows


# In[8]:


# Deleting rows with missing values

df_cleaned = df.dropna()
df_cleaned.isna().any()


# In[28]:


# Checking for Duplicates
df_cleaned.duplicated(subset=None, keep='first') # No duplicates in data set


# In[ ]:





# In[ ]:





# In[10]:


# CHecking data types of columns

df_cleaned.dtypes


# In[9]:


# converting budget and gross columns into integers
df_cleaned['budget']=df_cleaned['budget'].astype('int64')
df_cleaned['gross']=df_cleaned['gross'].astype('int64')
df_cleaned


# In[11]:


# Ordering Data by Gross revenue, as I want to focus on this column
df_cleaned.sort_values(by=["gross"], inplace=False, ascending=False)


# In[12]:


# Showing entire dataset

pd.set_option("display.max_rows", None) #default max is set at 20, therefor none to show all rows
df_cleaned


# In[13]:


# Checking for duplicates in name column, as duplicates in other columns are not as important/ even possible

df_cleaned['name'].drop_duplicates().sort_values(ascending=False)


# In[43]:





# In[ ]:





# ## Different Hypothesises that I want to check:
# #### Budget has high correlation with gross earnings
# #### Company has high correlation with gross earnings
# ##### Director has high correlation with gross earnings
# ##### The rating (stars) has high correlation with gross earnings

# # Budget and Gross Earnings

# In[36]:


from scipy.stats import spearmanr
corr_size_weight_s, p_size_weight_s = spearmanr(df_cleaned['budget'], df_cleaned['gross']) #Quality Column has been changed to "Label"
print(f"Budget and Gross Revenue (Spearman): Correlation={corr_size_weight_s}, p-value={p_size_weight_s}")


# In[39]:


# Scatter Plot with budget vs gross

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_cleaned['budget'], y=df_cleaned['gross'], color="green")
plt.title('Budget vs. Gross Earnings', size=20) 
plt.xlabel("Gross Earnings", size=15)  
plt.ylabel("Budget for Film", size=15)   

sns.regplot(x='budget', y='gross', data=df_cleaned, scatter=False, color="red", line_kws={"linewidth": 2})

plt.show()


# ### Positive Correlation between Budget and gross earnings of 0.69 wit relevant p-value

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[47]:


# Creating simple correlation matrix
# .corr() will use pearson correlation

df_cleaned.corr()


# In[ ]:


# Next, I tested different correlation methods


# In[48]:


df_cleaned.corr(method='kendall')
#kendall correlation is lower with 0.51


# In[49]:


df_cleaned.corr(method='spearman')


# In[56]:


# Visualizing Pearson Correlation between budget and gross in form of correlation matrix

correlation_matrix = df_cleaned.corr(method='pearson')

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True)

plt.title('Pearson Correlation Matrix')
plt.xlabel("Movie Features")
plt.ylabel("Movie Features")

plt.show


# # Company and Gross Revenue Correlation

# In[59]:


# Since company names are not numeric, I converted them into numeric data

df_numerized = df_cleaned

for column_name in df_numerized.columns:
    if(df_numerized[column_name].dtype == "object"):
        df_numerized[column_name]=df_numerized[column_name].astype("category")
        df_numerized[column_name]=df_numerized[column_name].cat.codes
        
df_numerized.sort_values(by=["gross"], inplace=False, ascending=False)
        
df_numerized


# In[ ]:





# In[58]:


df_numerized.corr()


# In[24]:


# Turning Correlation Matrix into table form, to check individual correlations easier

correlation_matrix=df_numerized.corr()

corr_pairs = correlation_matrix.unstack()
corr_pairs


# In[25]:


# Sorting by values
sorted_pairs = corr_pairs.sort_values()

sorted_pairs


# # Verifying / Falsyfing Hoypothesises
# ### Budget and Gross Earnings: 0.74
# ### Company and Gross Earnings 0.14
# ### Director and Gross Earnings: -0.029
# ### Star and Gross Earnings: -0.000004
# 
# ## Only one of the hypothesises turned out to be true with a significant correlation
# ## A relevant correlation, that I did not cover in my hyptothesises, was the impact of votes on gross earnings, as can be seen below

# In[26]:


# Looking for the highest Correlations:

high_correlation=sorted_pairs[(sorted_pairs) > 0.5]
high_correlation

#Gross and Budget
#Gross Votes

# Votes and Budget have the highest correlation to Gross Earnings


# # Votes and Budget have the highest correlation to Gross Earnings

# In[ ]:





# In[ ]:





# In[ ]:




