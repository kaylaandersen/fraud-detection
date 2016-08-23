import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt
# from clean_data_combined import clean_df
# from kmeans_analysis import plot_elbow
import numpy as np
import pickle

def extract_html_documents(df,column_name):
    '''
    Input: Pandas dataframe and column to extract text from
    Output: array of text documents
    '''

    html_strs = df[column_name]
    documents = html_strs.apply(lambda x: BeautifulSoup(x).get_text())
    documents = documents.apply(lambda x: unicodedata.normalize("NFKD", x).strip().encode('utf-8'))
    return documents

def tokenization_lemmatization(documents):
    '''
    Input: Documents array
    Output: Array of tokenized and lemmatized documents
    '''
    tokens = documents.apply(lambda x: word_tokenize(x.translate(None, string.punctuation)))
    stop = set(stopwords.words('english'))
    docs = tokens.apply(lambda words: [word.lower() for word in words if word.lower() not in stop])
    wordnet = WordNetLemmatizer()
    docs_wordnet = docs.apply(lambda words: [wordnet.lemmatize(word.decode('utf-8')) for word in words])
    return docs_wordnet

def tfidf(docs):
    '''
    Input: Documents
    Output: TFIDF of documents
    '''
    tfidf = TfidfVectorizer(stop_words='english',max_features=2000)
    tfidfed = tfidf.fit_transform(docs)
    return tfidfed

def make_kmeans_model(tfidfed):
    km = KMeans(n_clusters=11, init='k-means++', n_init=1,verbose=True, random_state=10)
    km.fit(tfidfed)
    return km

if __name__=='__main__':
    df = pd.read_json('../data/train_new.json')
    df = clean_df(df, drop_strings=False)
    tfidfed = tfidf(extract_html_documents(df,'description'))
    km = make_kmeans_model(tfidfed)
