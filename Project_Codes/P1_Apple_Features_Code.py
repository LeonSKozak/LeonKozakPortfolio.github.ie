#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# In[3]:


Appledf = pd.read_csv('/Users/leonkozak/Documents/DA_Marketing/apple_quality_copy.csv')


# In[4]:


Appledf.head()


# In[5]:


Appledf.info()


# ### Cleaning 

# In[6]:


Appledf.isna().any()
#There are missing values in all rows besides acidity in row 4002.


# In[7]:


Appledf.dropna(inplace=True)


# In[8]:


Appledf.info()
#Row with missing values has been deleted


# ### Duplicates

# In[9]:


Appledf.duplicated(subset=None, keep='first')
#No duplicates in data set


# In[10]:


Appledf.duplicated().sum()
#No duplicates in data set


# In[11]:


for column in Appledf.columns:
    num_distinct_values = len(Appledf[column].unique())
    print(f"{column}: {num_distinct_values} distinct values")


# In[12]:


# Only 2 distinct values in Quality Column, as there is only "good" and "bad" quality


# ### Encoding

# In[13]:


#encoding Quality feature into Label with values 0=good quality and 1= bad quality
def clean_data(Appledf):
    
    Appledf = Appledf.drop(columns=['A_id'])
    
    Appledf = Appledf.dropna()
    
    Appledf = Appledf.astype({'Acidity': 'float64'})
    
    def label(Quality):
        """
        Transform based on the following examples:
        Quality    Output
        "good"  => 0
        "bad"   => 1
        """
        if Quality == "good":
            return 0
    
        if Quality == "bad":
            return 1
    
        return None
    
    Appledf['Label'] = Appledf['Quality'].apply(label)
    
    Appledf = Appledf.drop(columns=['Quality'])
    
    Appledf = Appledf.astype({'Label': 'int64'})
    
    return Appledf

Appledf_clean = clean_data(Appledf.copy())
Appledf_clean.head()


# ### Checking Distribution of data

# In[16]:


numerical_cols = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']

plt.figure(figsize=(20, 10)) 
sns.set_palette(["red", "green"])

for i, column in enumerate(numerical_cols, 1):
    plt.subplot(2, 4, i)
    sns.histplot(data=Appledf_clean, x=column, kde=True, bins=20, color="red" if i % 2 == 0 else "green")
    plt.title(column, size=18)

plt.tight_layout()

plt.show()


# In[24]:


import pandas as pd

numerical_cols = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity', 'Label']

# Calculating skewness 
skewness = Appledf_clean[numerical_cols].skew()

# Calculating kurtosis 
kurtosis = Appledf_clean[numerical_cols].kurtosis()

print("Skewness:\n", skewness)
print()
print("Kurtosis:\n", kurtosis)


# ### Outliers

# In[17]:


from sklearn.preprocessing import RobustScaler, StandardScaler

numerical_features = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness',
       'Acidity']

robust_scaler = RobustScaler()

Appledf_clean[numerical_features] = robust_scaler.fit_transform(Appledf_clean[numerical_features])

def count_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).sum()

for feature in numerical_features :
    num_outliers = count_outliers(Appledf_clean[feature])
    print(f'Number of outliers in {feature}: {num_outliers}')


# ### Correlation Matrix

# In[20]:


correlation_matrix = Appledf_clean.corr()
columns = Appledf_clean.columns

p_value_matrix = pd.DataFrame(data=np.zeros_like(correlation_matrix, dtype=float), columns=columns, index=columns)

for i, row in enumerate(columns):
    for j, col in enumerate(columns):
        if i != j:
            _, p_value = pearsonr(Appledf_clean[row], Appledf_clean[col])
            p_value_matrix.iloc[i, j] = p_value
        else:
            p_value_matrix.iloc[i, j] = np.NaN

# combined matrix that also shows p-values
combined_matrix = [[f"{correlation_matrix.iloc[i, j]:.2f}\n(p={p_value_matrix.iloc[i, j]:.2g})"
                 if not np.isnan(correlation_matrix.iloc[i, j]) else ""
                 for j in range(len(columns))] for i in range(len(columns))]

sns.set(style="white", font_scale=1)

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=combined_matrix, fmt='', cmap='Blues', linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix of Apple Features with P-Values', size=24)

plt.tight_layout()

plt.savefig('/Users/leonkozak/Documents/DA_Marketing/Matrix_Blue.jpg', format='jpg', dpi=300)

plt.show()


# ### Story 1: Correlation between Weight and Size

# In[28]:


# Size and Weight
from scipy.stats import spearmanr
corr_size_weight_s, p_size_weight_s = spearmanr(Appledf_clean['Size'], Appledf_clean['Weight'])
print(f"Size and Weight (Spearman): Correlation={corr_size_weight_s}, p-value={p_size_weight_s}")


# In[26]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Weight', y='Size', data=Appledf_clean, color="green")
plt.title('Weight vs. Size', size=20)  
plt.xlabel('Weight', size=15)  
plt.ylabel('Size', size=15)    

#Trend Line:
sns.regplot(x='Weight', y='Size', data=Appledf_clean, scatter=False, color="red", line_kws={"linewidth": 2})

plt.show()


# ### Story 2: Correlation between Ripeness and Sweetness

# In[29]:


corr_size_weight_s, p_size_weight_s = spearmanr(Appledf_clean['Ripeness'], Appledf_clean['Sweetness'])
print(f"Ripeness and Sweetness (Spearman): Correlation={corr_size_weight_s}, p-value={p_size_weight_s}")


# In[35]:


import pandas as pd
import matplotlib.pyplot as plt

# Tend Line:
m, b = np.polyfit(Appledf_clean['Ripeness'], Appledf_clean['Sweetness'], 1)

# This time, I decided to do a hexbin plot
plt.figure(figsize=(10, 6))
hb=plt.hexbin(Appledf_clean['Ripeness'], Appledf_clean['Sweetness'], gridsize=30, cmap='Blues', mincnt=1)
plt.colorbar(hb, label='Count in bin')

plt.plot(Appledf_clean['Ripeness'], m*Appledf_clean['Ripeness'] + b, color='red', linewidth=2)

plt.title('Ripeness vs Sweetness of Apples')
plt.xlabel('Ripeness')
plt.ylabel('Sweetness')

plt.show()


# ### Story 3: Ripeness vs Quality

# In[36]:


corr_size_weight_s, p_size_weight_s = spearmanr(Appledf_clean['Ripeness'], Appledf_clean['Label']) #Quality Column has been changed to "Label"
print(f"Ripeness and Quality (Spearman): Correlation={corr_size_weight_s}, p-value={p_size_weight_s}")


# In[37]:


sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))

#Here, I decided to use a boxplot
sns.boxplot(x='Label', y='Ripeness', data=Appledf_clean, palette=["lightgreen", "pink"])
plt.title('Ripeness Distribution by Apple Quality', size=20)
plt.xlabel('Quality (Label)', size=15)
plt.ylabel('Ripeness', size=15)
plt.xticks([0, 1], ['Good', 'Bad'])

plt.show()


# #### As expected, Ripeness matters when it comes to Quality.
# #### Since apples are climacteric fruits, meaning that they continue to ripe after harvest, they should not be overly ripe when they are harvested.

# ### Story 4: Juiciness vs Acidity

# In[38]:


corr_size_weight_s, p_size_weight_s = spearmanr(Appledf_clean['Juiciness'], Appledf_clean['Acidity']) #Quality Column has been changed to "Label"
print(f"Juiciness and Acidity (Spearman): Correlation={corr_size_weight_s}, p-value={p_size_weight_s}")


# In[42]:


plt.figure(figsize=(10, 6))

#Again, I choose a scatterplot
sns.scatterplot(x='Juiciness', y='Acidity', data=Appledf_clean, color="red")
plt.title('Juiciness vs. Acidity', size=20)
plt.xlabel('Juiciness', size=15)
plt.ylabel('Acidity', size=15)
# Trend Line:
sns.regplot(x='Juiciness', y='Acidity', data=Appledf_clean, scatter=False, color="green", line_kws={"linewidth": 2})

plt.show()


# #### This shows us, that the jucier and apple is, the higher its acidity levels are. 
# #### Why could this be? 
# #### The acids responsible for the sourness in apples are more effectively perceived with higher water content, which helps to disseminate acidity across the taste buds.

# ### Story 5: Which apple feature is most important for quality?

# In[44]:


import matplotlib.pyplot as plt
import pandas as pd

# Again, I am calculating the correlations of all features with the quality label.
# Therefore, I am removing the quality label, so that i dont calculate the correlation between quality and quality
correlations_with_label = Appledf_clean.corr()['Label'].drop('Label') 

features = correlations_with_label.index
correlation_values = correlations_with_label.values

# Splitting data into 2 parts:
# positive correlation (bad quality)
positive_correlations_values = [-abs(value) for value in correlation_values if value > 0]  # positive correlations must be negative for plotting to left
# negative correlation (good quality) 
negative_correlations_values = [abs(value) for value in correlation_values if value < 0]  # negative correlations must be pos for plotting to right

positive_correlations_features = [features[i] for i, value in enumerate(correlation_values) if value > 0]
negative_correlations_features = [features[i] for i, value in enumerate(correlation_values) if value < 0]

# Plotting mirror chart
plt.figure(figsize=(12, 8))

# positive correlations on the left
plt.barh(positive_correlations_features, positive_correlations_values, color='red', label='Bad Quality (Positive Correlation)')

# negative correlations on the right
plt.barh(negative_correlations_features, negative_correlations_values, color='green', label='Good Quality (Negative Correlation)')

plt.title('Feature Correlation with Apple Quality: Good vs. Bad Quality')
plt.xlabel('Correlation with Quality Label')
plt.ylabel('Features')
plt.legend(loc='lower right')

# vertical line to show x=0
plt.axvline(x=0, color='grey', linestyle='--')
plt.show()


# #### The Bar Chart shows the most important features for predicting Apple Quality according to our Data Set
# #### The most important features to get a good quality apple are: High Juiciness, High Sweetness and Large Size
# #### The most important features to get a bad quality apple are: Low Ripeness
# #### The Features Weight, Acidity and Crunchiness do not have a big impact on apple quality

# ### The Best Apple within the data set according to the calculated features

# In[47]:


# I used the top and worst quartiles to find the best apple. 
# It had to score ery high in Juiciness, Sweetness and Size
top_quartile_thresholds = {
    'Juiciness': Appledf_clean['Juiciness'].quantile(0.75),
    'Sweetness': Appledf_clean['Sweetness'].quantile(0.75),
    'Size': Appledf_clean['Size'].quantile(0.75),
}

# The best apple has to score low in Ripeness
bottom_quartile_thresholds = {
    'Ripeness': Appledf_clean['Ripeness'].quantile(0.25),
}

best_apples_criteria = (Appledf_clean['Juiciness'] >= top_quartile_thresholds['Juiciness']) & \
                       (Appledf_clean['Sweetness'] >= top_quartile_thresholds['Sweetness']) & \
                       (Appledf_clean['Size'] >= top_quartile_thresholds['Size']) & \
                       (Appledf_clean['Ripeness'] <= bottom_quartile_thresholds['Ripeness'])

best_apples = Appledf_clean[best_apples_criteria]

# Calculating a combined score 
best_apples['Combined_Score'] = best_apples[['Juiciness', 'Sweetness', 'Size']].sum(axis=1) - best_apples['Ripeness']

# Next, I looked for the apple with the highest combined score
best_apple_index = best_apples['Combined_Score'].idxmax()
best_apple = best_apples.loc[best_apple_index]

best_apple


# #### This (2532) is the best apple in the data set

# In[50]:


# Here, I had problems, so I entered the correct calculated values by hand
best_apple_correct_values = {
    'Size': 4.842414,
    'Sweetness': 1.737140,
    'Juiciness': 5.288582,
    'Ripeness': -0.967865  # Note on negative value
}

# Creating series for plotting
best_apple_correct_series = pd.Series(best_apple_correct_values, name='Feature Scores')

plt.figure(figsize=(10, 6))
best_apple_correct_series.plot(kind='barh', color='skyblue')
plt.title('Best Apple (2532) Feature Scores')
plt.xlabel('Scores')
plt.ylabel('Features')

for index, value in enumerate(best_apple_correct_series):
    plt.text(value, index, f"{value:.2f}", va='center')

plt.tight_layout()
plt.show()


# ### The Worst Apple  within te data set according to the calculated features:

# In[52]:


# Just logic like finding the best apple
bottom_quartile_thresholds_worst = {
    'Juiciness': Appledf_clean['Juiciness'].quantile(0.25),
    'Sweetness': Appledf_clean['Sweetness'].quantile(0.25),
    'Size': Appledf_clean['Size'].quantile(0.25),
}

top_quartile_thresholds_worst = {
    'Ripeness': Appledf_clean['Ripeness'].quantile(0.75),
}


worst_apples_criteria = (Appledf_clean['Juiciness'] <= bottom_quartile_thresholds_worst['Juiciness']) & \
                        (Appledf_clean['Sweetness'] <= bottom_quartile_thresholds_worst['Sweetness']) & \
                        (Appledf_clean['Size'] <= bottom_quartile_thresholds_worst['Size']) & \
                        (Appledf_clean['Ripeness'] >= top_quartile_thresholds_worst['Ripeness'])

worst_apples = Appledf_clean[worst_apples_criteria].copy()

worst_apples['Combined_Score_Worst'] = -(worst_apples[['Juiciness', 'Sweetness', 'Size']].sum(axis=1)) + worst_apples['Ripeness']

worst_apple_index = worst_apples['Combined_Score_Worst'].idxmax()
worst_apple = worst_apples.loc[worst_apple_index]

worst_apple[['Size', 'Juiciness', 'Sweetness', 'Ripeness', 'Combined_Score_Worst']]


# #### This (1712) is the worst apple in the data set

# In[53]:


# Again, I entered the features by hand
worst_apple_scores = {
    'Size': worst_apple['Size'],
    'Juiciness': worst_apple['Juiciness'],
    'Sweetness': worst_apple['Sweetness'],
    'Ripeness': worst_apple['Ripeness']
}

# Again, I created a series for plotting
worst_apple_series = pd.Series(worst_apple_scores, name='Feature Scores')

plt.figure(figsize=(10, 6))
worst_apple_series.plot(kind='barh', color='salmon')
plt.title(f'Worst Apple ({worst_apple_index}) Feature Scores')
plt.xlabel('Scores')
plt.ylabel('Features')

for index, value in enumerate(worst_apple_series):
    plt.text(value, index, f"{value:.2f}", va='center')

plt.tight_layout()
plt.show()


# ### Here, I calculated which features where the most important using random forrests

# In[56]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Again, dropping qualiyt label
X = Appledf_clean.drop(columns=['Label'])

# Target variable is quaity (label)
y = Appledf_clean['Label']

# Splitting into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RF_Model = GradientBoostingClassifier(n_estimators=100, random_state=42)
RF_Model.fit(X_train, y_train)

# Extract and sort feature importances
importances_RF = RF_Model.feature_importances_
features = X.columns
indices_FM = np.argsort(importances_RF)[::-1]

# Display the sorted feature importances
sorted_features_RF = [(features[i], importances_RF[i]) for i in indices_FM]
sorted_features_RF


# In[58]:


feature_names = [name for name, score in sorted_features_RF]
importance_scores = [score for name, score in sorted_features_RF]

plt.figure(figsize=(12, 8))
plt.barh(feature_names, importance_scores, color='skyblue')
plt.xlabel('Importance Score')
plt.title('Feature Importance for Apple Quality Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# #### According to the random forrest model, similar features play a role in predicting apple quality. However, the rf model does not show if these features must be negative or positive

# In[ ]:




