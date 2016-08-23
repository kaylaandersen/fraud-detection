from clean_data import clean_bottom
import pandas as pd
from exploration import clean_top

USE_POINT = '../data/test_point.json'

def load_point(file_name):
    test = pd.read_json(file_name, typ='series')
    return test

if __name__ == '__main__':
    # Now have a series
    test = load_point(USE_POINT)
    test_df = pd.DataFrame(test).T

    test_df = clean_top(test_df)
    test_df = clean_bottom(test_df)
