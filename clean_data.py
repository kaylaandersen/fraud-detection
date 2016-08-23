import pandas as pd
import numpy as np
from exploration import clean_top

# Constant
USE_DATA    = '../data/train_new.json'
TEST_POINT  = '../data/test_point.json'
TEST_INDEX  = 300
BAD_VALUES  = [np.nan]

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


def func_with_error(func, dct, key):
    """
    DESCR: needed for lambda min and max on empty lists
    INPUT:
        func - func that fails on empty lists
        dct - dct to use in comphrehension
        key - key from dct for list comphrehension
    OUTPUT:
        func of list comp or 0
    """
    try:
        return func([dct[key] for key in dct])
    except:
        return 0


def clean_bottom(df):
    """
    DESCR: Deals with bottom portion of dataframe data cleaning
    INPUT:
        df: DataFrame, messy df
    OUTPUT:
        df: DataFrame, clean df
    """
    # Drop columsn that won't exist in test point


    # Handle venue_state info
        # make a column to specify if not given, then drop venue state
    df['no_venue'] = df['venue_state'].apply(lambda x: 1 if x in BAD_VALUES else 0)

    # Handle venue_country
    df['venue_user_country_mismath'] = df['venue_country'] != df['country']
    df['venue_user_country_mismath'] = df['venue_user_country_mismath'].apply(lambda x: 1 if x else 0)

    # Handle venue address
    df['no_address'] = df['venue_address'].apply(lambda x: 1 if x is "" else 0)
    df['no_venue name'] = df['venue_name'].apply(lambda x: 1 if x is "" else 0)
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


    return df

def run_clean_data():
    raw = load_data(USE_DATA)
    # raw.iloc[TEST_INDEX].to_json(TEST_POINT)
    df = add_fraud_col(raw)
    df = clean_top(df)
    df = clean_bottom(df)
    return df
