#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[7]:


import pandas as pd
data=pd.read_csv("Elon_musk (1).csv",encoding='unicode_escape',error_bad_lines=False) 
data.head()
d1 = data.iloc[:,1]
d1


# In[8]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[9]:


string.punctuation


# In[83]:


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

d2 = d1.apply(cleanTxt)

d2


# In[61]:


d3 = d2.str.translate(str.maketrans('', '', string.punctuation))
d3


# In[62]:


from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(str(d3))
print(text_tokens[0:50]) 


# In[53]:


len(text_tokens) 


# In[63]:


import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

my_stop_words = stopwords.words('english')
my_stop_words.append('the')
no_stop_tokens = [word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens[0:40])


# In[64]:


lower_words = [x.lower() for x in no_stop_tokens]
print(lower_words[0:25]) 


# In[65]:


#Stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in lower_words]
print(stemmed_tokens[0:40]) 


# In[66]:


# NLP english language model of spacy library
nlp = spacy.load('en_core_web_sm')   


# In[67]:


# lemmas being one of them, but mostly POS, which will follow later
doc = nlp(' '.join(no_stop_tokens))
print(doc[0:40]) 


# In[68]:


lemmas = [token.lemma_ for token in doc]
print(lemmas[0:25])


# ## Feature Extraction

# In[69]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(lemmas) 


# In[70]:


print(vectorizer.vocabulary_)


# In[71]:


print(vectorizer.get_feature_names()[50:100])
print(X.toarray()[50:100]) 


# In[72]:


print(X.toarray().shape) 


# In[73]:


vectorizer_ngram_range = CountVectorizer(analyzer='word',ngram_range=(1,3),max_features = 100)
bow_matrix_ngram =vectorizer_ngram_range.fit_transform(d3) 


# In[74]:


print(vectorizer_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# In[75]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_n_gram_max_features = TfidfVectorizer(norm="l2",analyzer='word', ngram_range=(1,3), max_features = 500)
tf_idf_matrix_n_gram_max_features =vectorizer_n_gram_max_features.fit_transform(d3)
print(vectorizer_n_gram_max_features.get_feature_names())
print(tf_idf_matrix_n_gram_max_features.toarray()) 


# In[76]:


# Import packages
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off"); 


# In[82]:


# Generate wordcloud
stopwords = STOPWORDS
stopwords.add('will')
wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(str(d3))
# Plot
plot_cloud(wordcloud)

