#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk #natural language toolkit, needed for sentiment analysis


# In[6]:


df = pd.read_csv('/Users/leonkozak/Documents/Reviews.csv')
df.head()


# In[7]:


print(df.shape)
df=df.head(250) #downsizing data set to 250, to make it more managable


# In[8]:


ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars', figsize=(10,5))

ax.set_xlabel('Review Stars')
plt.show()

# Many positive Reviews in Data Set
# Very biased towards positive reviews


# # NLTK

# In[9]:


example = df['Text'][50]
print(example)


# In[10]:


# NLTK is able of tokenizing each word within the Reviews

tokens=nltk.word_tokenize(example)
tokens[:10]


# In[11]:


# NLTK can find part of speech for each word
# .pos = part of speech

tagged=nltk.pos_tag(tokens) #each tokens has a part of speech value (NN= singular noun etc.)
tagged[:10]


# In[12]:


# Tagged parts of speech can be put into entities

entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# # Vader Sentiment Scoring

# In[13]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
import nltk

sia = SentimentIntensityAnalyzer()
#This Sentiment Intensity analyzer can now be run on text to detect the sentiment of the given text.


# In[14]:


sia.polarity_scores('I am so happy!') #Comment is mostly positive


# In[15]:


sia.polarity_scores('This is the worst thing ever.') # mostly negative and neutral, but not positive


# In[16]:


sia.polarity_scores(example) #example is overall negative/neutral


# ### Running Polarity Score on the entire data set.

# In[17]:


res = {}

for i, row in tqdm(df.iterrows(), total = len(df)): #total is length of data frame --> X/568454
    text = row['Text'] #Text column
    myid = row['Id'] #Id column
    res[myid] = sia.polarity_scores(text) #storing results in myid part of dictionary
    


# In[18]:


res #each id with its, neg, neu, pos and compound score


# In[19]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')


# In[20]:


#turning res into pandas data frame
vaders = vaders.drop_duplicates(subset='Id', keep='first') #deleting duplicates from vader to proceed

vaders = pd.DataFrame(res).T #.T to orient it horizontaly
vaders


# In[21]:


#Merging Vaders onto original Data Frame

vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how = 'left')


# In[22]:


# Now sentiment score and meta data (vader) merged with DataFrame
vaders.head()


# In[23]:


#Assumption: The higher the score, the more positive the polarity score


# In[24]:


#plotting barplot to see if Assumption is correct

sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()

#Assumption was correct


# In[25]:


fig, axs = plt.subplots(1,3, figsize=(15,5))

sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')

plt.tight_layout()

plt.show()


# # Roberta Pretrained Model

# In[26]:


# Use a model trained of a large corpus of data
# Transformer model accounts for the words but also the context related to other words


# In[27]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[28]:


# Downloading a pre-trained tokenizer that was trained on Twitter Data

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[29]:


# Vader Results on example

print(example)
sia.polarity_scores(example)


# In[30]:


# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)


# In[31]:


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict


# In[32]:


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')


# In[36]:


results_df=pd.DataFrame(res).T
results_df=results_df.reset_index().rename(columns={'index':'Id'})
results_df=results_df.merge(df, how='left')


# In[37]:


results_df.head()


# ## Comparing Scores Between Models

# In[39]:


results_df.columns


# In[40]:


sns.pairplot(data=results_df, vars=['vader_neg', 'vader_neu', 'vader_pos', 'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score', palette='tab10')
plt.show()


# ## Review Examples

# In[44]:


# Checking for wrong sentiment ratings in Roberta
# Positive 1 star review and negative 5 star reviews

results_df.query('Score == 1') \
.sort_values('roberta_pos', ascending=False)['Text'].values[0]


# In[45]:


# Checking for wrong sentiment ratings in Vader
# Positive 1 star review and negative 5 star reviews

results_df.query('Score == 1') \
.sort_values('vader_pos', ascending=False)['Text'].values[0]


# In[46]:


# Negative Sentiment 5 Star review
# Roberta
results_df.query('Score == 5') \
.sort_values('roberta_neg', ascending=False)['Text'].values[0]


# In[47]:


# Negative Sentiment 5 Star review
# Vader
results_df.query('Score == 5') \
.sort_values('vader_neg', ascending=False)['Text'].values[0]

