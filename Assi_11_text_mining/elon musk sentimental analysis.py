#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string # special operations on strings
import spacy # language models

from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud
get_ipython().run_line_magic('matplotlib', 'inline')
import re
import string


# In[2]:


import pandas as pd
data=pd.read_csv("Elon_musk (1).csv",encoding='unicode_escape',error_bad_lines=False) 
data.head()
data.drop('Unnamed: 0',inplace=True,axis=1)
data


# In[3]:


string.punctuation


# In[74]:


def cleanTxt(text):
    text = re.sub(r'\\u[A-Za-z0-9]+','',text)
    text = re.sub(r'@[A-Za-z0-9]+','',text)
    text = re.sub(r'https?:\/\/\S+','',text)
    text = re.sub('[0-9]', '', text)
    text = re.sub(r'\s+',' ', text)
    text = re.sub('[_AA_]','',text)
    text = re.sub('[<U+F>]','',text)
    text = re.sub('[&]','',text)
    text = re.sub('[;,?]','',text)
    return text

data['Text'] = data['Text'].apply(cleanTxt)

data


# In[75]:


nlp = spacy.load('en_core_web_sm')


# In[65]:


one_block = data.Text[35]
doc_block = nlp(one_block)
spacy.displacy.render(doc_block, style='ent', jupyter=True) 


# In[76]:


one_block


# In[77]:


for token in doc_block[0:30]:
    print(token, token.pos_) 


# In[78]:


#Filtering for nouns and verbs only
nouns_verbs = [token.text for token in doc_block if token.pos_ in ('NOUN', 'VERB')]
print(nouns_verbs[5:30]) 


# In[79]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(nouns_verbs)
sum_words = X.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
wf_df = pd.DataFrame(words_freq)
wf_df.columns = ['word', 'count']

wf_df[0:35] 


# In[80]:


#Sentiment analysis
affin = pd.read_csv('Afinn.csv', sep=',', encoding='latin-1')
affin


# In[81]:


affin.head()


# In[87]:


from nltk import tokenize
sentences = tokenize.sent_tokenize(" ".join(data.Text))
sentences[0:30] 


# In[88]:


sent_df = pd.DataFrame(sentences, columns=['Text'])
sent_df


# In[89]:


affinity_scores = affin.set_index('word')['value'].to_dict() 


# In[90]:


#Custom function :score each word in a sentence in lemmatised form, 
#but calculate the score for the whole original sentence.
nlp = spacy.load('en_core_web_sm')
sentiment_lexicon = affinity_scores

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0)
    return sent_score 


# In[34]:


calculate_sentiment(text = 'amazing') 


# In[37]:


sent_df['sentiment_value'] = sent_df['Text'].apply(calculate_sentiment)


# In[38]:


# how many words are in the sentence?
sent_df['word_count'] = sent_df['Text'].str.split().apply(len)
sent_df['word_count'].head(10) 


# In[39]:


sent_df 


# In[40]:


sent_df.sort_values(by='sentiment_value').tail(10) 


# In[41]:


sent_df['sentiment_value'].describe() 


# In[42]:


# Sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0].head() 


# In[46]:


sent_df[sent_df['sentiment_value']>=18].head() 


# In[47]:


sent_df['index']=range(0,len(sent_df)) 


# In[48]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(sent_df['sentiment_value']) 


# In[49]:


plt.figure(figsize=(15, 10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df) 


# In[50]:


sent_df.plot.scatter(x='word_count', y='sentiment_value', figsize=(8,8), title='Sentence sentiment value to sentence word count')

