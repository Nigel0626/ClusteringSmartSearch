#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
import nltk

stop_words = stopwords.words('english')

def clr_stop_words(words):  ##clear stop words
    filter_words = []
    for w in words:
        if w not in stop_words:
            filter_words.append(w)

    return filter_words 

def save_model(model, fileName):
    # Save Model to file in the current working directory
    joblib.dump(model, fileName)
    
def load_model(fileName):    
    # Load from file
    joblib_model = joblib.load(fileName)
    return joblib_model

def cleanDescriptions(text):
    text = text.lower()
    text = re.sub('\n', ' ',text)# clean extra line
    text = re.sub('terms and condition.*$', '', text) 
    #text = re.sub('terms and condition*', '', text)
    #m = re.search('terms and condition.*', text)
    #if m:
    #    text = m.group(0)
    #re.match('terms and condition', text).group(0)
    return text

def clean_data(data):
    msg = data.lower() #lower all charracter
    #msg = deEmojify(lower_msg) ##clear emoji
    msg = re.sub('[^a-zA-Z0-9.]+', '  ', msg) ##clean symbol
    msg = re.sub('(.x?)http.*?(.*?)', ' ', msg) #clean url
    word_token = nltk.tokenize.word_tokenize(msg)
    #msg = spell_checker(word_token)
    word_token = clr_stop_words(word_token)
    #lemmatized = lemma_words(clean_stop_words)
    text = ' '.join(map(str, word_token))
    return text


# In[21]:


model = load_model("svm_model.pkl")
vectorizer_ngram = load_model("vectorizer_ngram3.pkl")


# In[8]:


englishDFClean = pd.read_csv("Result2.csv")


# In[4]:


englishDFClean


# In[26]:


import numpy as np


def prediction(text, model, vectorizer,csv_file):
    englishDFClean = pd.read_csv(csv_file)
    testing = [text]
    test_vec = vectorizer.transform(testing)
    #print(test_vec)
    y_pred = model.predict(test_vec)
    y_pred = y_pred[0]

    rslt_df = englishDFClean[englishDFClean['Cluster'] == y_pred] 
    corpus = rslt_df['Cleaned_Spec+Desc']

    corpus = corpus.tolist()
    corpus.append(text)

    vect = TfidfVectorizer(min_df=1, stop_words="english",ngram_range =(1,3))                                                                                                                                                                                                   
    tfidf = vect.fit_transform(corpus)                                                                                                                                                                                                                       
    pairwise_similarity = tfidf * tfidf.T 

    pairwise_similarity.toarray()  
                                                                                                                                                                                                                                  
    arr = pairwise_similarity.toarray()     
    np.fill_diagonal(arr, np.nan)                                                                                                                                                                                                                            
                                                                                                                                                                                                                 
    #input_doc = "chargers power supply specifications adapter power supply input ac 100 240v50 60hz cable length 90cm 140cm 300em output adaptor 5.5mm x 2.1mm 2.5mm . plug type uk 3 pin polarity inner positive outer negative led light indicator colour black features compact size high reliability light weight . professional dc112v1a 12v2a 12v3a 9v1a 9v2a 5v2a 6v2a switching power supply adapter high quality materials durable usage voltage protection load protection short circuit protection perfect power supply external hard drive wireless router cctv camera tv box ete . perfect power supply adapter led strip light stage lighting portable external hard drive wireless router cctv surveillance camera adsl modem mini tv mobile dvd tv box audio video equipments charging equipment mp3 mp4 small table lamp game console telephone lantern cameras surveillance equipment controllers home portable devices . please read purchasing note please make sure dc output connector size compatible device bidding . compatible charging electric toy car bike motor . indicator light change color accept returns exchanges unless item purchased defective . package include 1 x ac adapter cctv decoder dvbt2k2 adapter sv adapter 12v andriod adapterv"                                                                                                                                                                                                  
    input_idx = corpus.index(text)                                                                                                                                                                                                                      
    idx = (-arr[input_idx]).argsort()[:5]
    
    print("Similar Product")
    print("==============================================================================")
    for i in idx:
        print(rslt_df.iloc[i]['Cleaned_Name'], "\nConfidence Score: ", round(arr[input_idx][i],3))
        print("\n")
        
text = input('Enter the description:')
text = cleanDescriptions(text)
text = clean_data(text)
prediction(text, model, vectorizer_ngram, "Result2.csv")

