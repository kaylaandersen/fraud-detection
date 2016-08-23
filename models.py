import clean_data_combined
import pandas as pd
import numpy as np
from build_models import Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from Grid_Search_Params import grid_search_params

def oversample(X, y, target=0.15):
    if target < sum(y)/float(len(y)):
        return X,y
    positive_count = sum(y)
    negativ_count = len(y) - positive_count
    target_positive_count = target*negativ_count / (1. - target)
    target_positive_count = int(round(target_positive_count))
    number_of_new_observations = target_positive_count - positive_count

    positive_obs_indices = np.where(y==1)[0]
    new_obs_indices = np.random.choice(positive_obs_indices,
                                        size=number_of_new_observations,
                                        replace=True)
    X_new, y_new = X[new_obs_indices], y[new_obs_indices]
    X_positive = np.vstack((X[positive_obs_indices], X_new))
    y_positive = np.concatenate((y[positive_obs_indices], y_new))
    X_negative = X[y==0]
    y_negative = y[y==0]
    X_oversampled = np.vstack((X_negative, X_positive))
    y_oversample = np.concatenate((y_negative, y_positive))

    return X_oversampled,y_oversample

if __name__ == '__main__':

    #pull, clean and split data
    data = clean_data_combined.run_clean_data()
    y = data.pop('fraud').values
    x = data.values
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=.70)
    updated_x_train, updated_y_train = oversample(x_train,y_train)

    #Test Logic Reg
    rf = Model(RandomForestClassifier,grid_search_params['RandomForestClassifier'])
    rf.run_grid_search(updated_x_train,updated_y_train)
    y_pred = rf.predict_model(x_test)
    print rf.score(y_test)
