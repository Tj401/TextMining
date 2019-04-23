# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:09:37 2019

@author: kdandebo
"""

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', 500)
df = pd.ExcelFile('C:/Users/kdandebo/Desktop/HomelatoptoKarthiklaptop/Python/datasetforpractice/sms_raw_NB.xlsx')
df = df.parse("sms_raw_NB")
print(df.head(10))
print(df.columns)
type(df)
#str(df['type'])
df['type'].describe()
df['text'].describe()
#df['type'].info()

#counting the number of words

df['wrdcount'] = df['text'].apply(lambda x: len(str(x).split(" ")))
#print(df['text']), print(wrdcount)
#print("geeks"), print("geeksforgeeks")

#print(df, end=" ")

df[['text','wrdcount']].head()


#counting the number of stop words
from nltk.corpus import stopwords
stop = stopwords.words('english')

df['stopwords'] = df['text'].apply(lambda x: len([x for x in str(x).split() if x in stop]))
df[['text','stopwords']].head()


#Caluculating the Number of hashtag characters

#df['splchr'] = df['text'].apply(lambda x: len([x for x in str(x).split() if x.startswith('#')]))
#df[['text','splchr']].head()


#caluculuating the number of numerics
#df['numerics'] = df['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
#df[['text','numerics']].head()

#Number of uppercase words
#df['upper'] = df['text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
#df[['text','upper']].head()


#****Cleaning of the corpus*******

#Converting evrthing to lowercase

df_clean = df['text']
df_clean.head(10)
df_clean.columns = ['text']
df_clean.columns

#convert to lowercase
df_clean = df['text'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
df_clean.head(10)

#Removing punctuation
df_clean = df_clean.str.replace('[^\w\s]','')
df_clean.head(10)

#Removal of stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
df_clean = df_clean.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop))
df_clean.head(10)

#Common words removal

#freq words
freq = pd.Series(' '.join(df_clean).split()).value_counts()[:10]
freq

#removal of freq words
freq = list(freq.index)
df_clean = df_clean.apply(lambda x: " ".join(x for x in str(x).split() if x not in freq))
df_clean.head(10)

#Do the spellcorrect

from textblob import TextBlob
df_clean[:5].apply(lambda x: str(TextBlob(x).correct()))
df_clean.head(10)

#removing all the suffcies by usng the stemming

from nltk.stem import PorterStemmer
st = PorterStemmer()
df_clean[:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
df_clean.head(10)


#lemmetization
#converts into root word 

from textblob import Word
df_clean = df_clean.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df_clean.head(10)


df.head(10)
