import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from NLP import extract_html_documents, tfidf, make_kmeans_model
import numpy as np
import pickle
# Constant
USE_DATA    = 'data/train_new.json'
TEST_POINT  = 'data/test_point.json'
TEST_INDEX  = 300

def load_data(file_name):
    """
    DESCR: get training data from json into df
    INPUT:
        filename: string, path to data in json form
    OUTPUT:
        df: DataFrame, df of json file
    """
    df = pd.read_json(USE_DATA)
    return df


def add_fraud_col(df):
    """
    DESCR: add a column based on ourd decided definition of fraud
    INPUT:
        df: DataFrame, df without fraud column
    OUTPUT:
        df: DataFrame, df w fraud column
    """
    df['fraud'] = df['acct_type'].apply(lambda x: 1 if 'fraud' in x else 0)
    return df


def clean_df(df, loss = False, drop_strings=True):
    """
    DESCR: Deals with bottom portion of dataframe data cleaning
    INPUT:
        df: DataFrame, messy df. Pass loss=True if you want the target variable to be scaled by the amount of money lost
    OUTPUT:
        df: DataFrame, clean df with all columns containing integers or float values.  Target is in df['fraud'].
    """

    df['approx_payout_date'] = pd.to_datetime(df['approx_payout_date'],unit='s')
    # most common countries = 1, else 0
    df['venue_user_country_match'] = df['venue_country'] == df['country']
    df['venue_user_country_match'] = df['venue_user_country_match'].astype(int)
    df['country'] = df['country'].apply(lambda x: 1 if x in ('US','GB','CA','AU','NZ') else 0)
    df['currency'] = df['currency'].apply(lambda x: 1 if x=='USD' else 2 if x=='EUR' else 3 if x=='CAD' else 4 if x=='GBP' else 5 if x=='AUD' else 6 if x=='NZD' else 7 if x=='MXN' else 8)
    df['delivery_method'] = df['delivery_method'].apply(lambda x: 5 if pd.isnull(x) else x).astype(int)
    df['description_char_length'] = df['description'].apply(lambda x: len(x))
    df['description_caps_pct'] =  df['description'].apply(lambda x: (sum(1 for y in x if y.isupper())+1)/float(len(x)+1))
    df['description_word_count'] = df['description'].apply(lambda x: len(x.split()))
    df['email_domain'] = df['email_domain'].apply(lambda x: x.upper())
    common_domains = df['email_domain'].value_counts().head(100).index.values
    df['common_email_domain'] = df['email_domain'].apply(lambda x: 1 if x in common_domains else 0)
    df['event_created'] = pd.to_datetime(df['event_created'],unit='s')
    df['event_end'] = pd.to_datetime(df['event_end'],unit='s')
    df['event_published'] = pd.to_datetime(df['event_published'],unit='s')
    df['event_published'][pd.isnull(df['event_published'])] = df['event_created'][pd.isnull(df['event_published'])]
    df['event_start'] = pd.to_datetime(df['event_start'],unit='s')
    # gts = gross total sales
    df['has_header'] = df['has_header'].apply(lambda x: 0 if pd.isnull(x) else x).astype(int)
    df['listed'] = df['listed'].apply(lambda x: 1 if x=='y' else 0)
    df['name_caps_pct'] =  df['name'].apply(lambda x: (sum(1 for y in x if y.isupper())+1)/float(len(x)+1))
    # objectID likely unnecessary
    # columns that need NLP techniques: description, name, org_desc
    df['org_facebook'] = df['org_facebook'].apply(lambda x: 100 if pd.isnull(x) else int(x))
    df['org_twitter'] = df['org_twitter'].apply(lambda x: 100 if pd. isnull(x) else x).astype(int)
    df['has_payee_name'] = df['payee_name'].apply(lambda x: 1 if x else 0)
    df['payout_type'] = df['payout_type'].apply(lambda x: 1 if x=='' else 2 if x=='CHECK' else 3 if x=='ACH' else 4)
    # previous_payouts contains address,amount,country,created,event,name,state,uid,zip_code
    df['num_prev_payouts'] = df['previous_payouts'].apply(lambda x: len(x))
    df['sum_prev_payouts'] = df['previous_payouts'].apply(lambda x:sum([payout['amount'] for payout in x]))
    df['num_payout_addresses'] = df['previous_payouts'].apply(lambda x: len(set([payout['address'] for payout in x])))
    #sale_duration contains 155 nulls
    df['sale_duration'] = df['sale_duration'].apply(lambda x: 0 if pd.isnull(x) else x).astype(int)
    #sale duration2 mostly matches up with sale_duration and doesnt have nulls, get null values from there instead of setting to 0


    # Handle venue address
    df['no_address'] = df['venue_address'].apply(lambda x: 1 if x == "" else 0)
    df['no_venue_name'] = df['venue_name'].apply(lambda x: 1 if (x == "" or x == None or x == 'None' or x == np.nan) else 0)
    # Handle user type
    # Handle user age

    # Handle user created
    df['user_created'] = pd.to_datetime(df['user_created'], unit='s')

    # Handle ticket ticket_types
    df['num_ticket_tiers'] = df['ticket_types'].apply(lambda x: len(x))
    df['max_ticket_cost'] = df['ticket_types'].apply(lambda x: max(  [ticket['cost'] for ticket in x] + [0]))
    df['min_ticket_cost'] = df['ticket_types'].apply(lambda x: min([ticket['cost'] for ticket in x] or [0]))
    df['total_tickets_allowed'] = df['ticket_types'].apply(lambda x: sum([ticket['quantity_total'] for ticket in x]))


    # Stuff to maybe drop
    df.drop('venue_state', inplace=True, axis=1)
    df.drop('venue_country', inplace=True, axis=1)
    df.drop('venue_longitude', inplace=True, axis=1)
    df.drop('venue_latitude', inplace=True, axis=1)
    df.drop('ticket_types', inplace=True, axis=1)
    df.drop('venue_address', inplace=True, axis=1)
    df.drop('venue_name', inplace=True, axis=1)
    df.drop('object_id', inplace=True, axis=1)

    # run kmeans cluster analysis on descriptions
    tfidfed = tfidf(extract_html_documents(df,'description'))
    f = open('kmeans_model.pkl')
    km = pickle.load(f)
    predicted = km.predict(tfidfed)
    df['description_nlp'] = predicted
    f.close()

    if drop_strings:
    # remove string columns (possibly use with NLP later)
        df.drop('description', inplace=True, axis=1)
        df.drop('email_domain', inplace=True, axis=1)
        df.drop('name', inplace=True, axis=1)
        df.drop('org_desc', inplace=True, axis=1)
        df.drop('org_name', inplace=True, axis=1)
        df.drop('payee_name', inplace=True, axis=1)
        df.drop('previous_payouts', inplace=True, axis=1)
        df.drop('acct_type', inplace=True, axis=1)

    # add time-interval column then drop datetime columns
    df['days_since_account_creation'] = (df['event_created']-df['user_created']).apply(lambda x: x.days)
    df.drop('user_created', inplace=True, axis=1)
    df.drop('event_start', inplace=True, axis=1)
    df.drop('event_published', inplace=True, axis=1)
    df.drop('event_end', inplace=True, axis=1)
    df.drop('event_created', inplace=True, axis=1)
    df.drop('approx_payout_date', inplace=True, axis=1)

    if loss:
        df['fraud'] = df['fraud'] * df['gts']
    return df

def run_clean_data():
    raw = load_data(USE_DATA)

    raw.iloc[TEST_INDEX].to_json(TEST_POINT)
    df = add_fraud_col(raw)
    df = clean_df(df)
    return df
