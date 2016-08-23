from clean_data import clean_bottom
import pandas as pd
from exploration import clean_top

def load_point(json_response):
    data_df = pd.read_json(json_response, typ='series')
    return data_df

def single_point_load(data_point):
    data_df = load_point(data_point)
    data_point_df = pd.DataFrame(data_df).T

    first_cleaned_df = clean_top(data_point_df)
    cleaned_df = clean_bottom(first_cleaned_df)
    return cleaned_df
