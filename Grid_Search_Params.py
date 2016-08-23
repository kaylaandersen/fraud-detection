
grid_search_params = {'DecisionTreeClassifier':{'criterion':['gini','entropy'], 'max_depth':[None,5,10,20], 'min_samples_split':[2,4,6], 'max_features':[None,3,7,10,15],'random_state':[1]},
'LogisticRegression':{'penalty':['l2','l1'], 'fit_intercept':[True,False], 'C':[0.0001,0.001,0.1,1],'random_state':[1], 'max_iter':[20,40,70,100],'n_jobs':[-1]},
'KNeighborsClassifier':{'n_neighbors':[5,10,20], 'weights':['uniform','distance'], 'p':[1,2], 'n_jobs':[-1]},
'SVC':{'C':[0.0001,0.001,0.1,1], 'kernel':['rbf','poly','sigmoid'], 'degree':[3,4,5,6,7,8], 'shrinking':[True,False],'random_state':[1]},
'RandomForestClassifier':{'n_estimators':[1000], 'criterion':['gini','entropy'],'max_depth':[None,5,20], 'min_samples_split':[2,6], 'max_features':[7], 'bootstrap':[True], 'n_jobs':[-1], 'random_state':[1],'verbose':[True]},
'GradientBoostingClassifier': {'learning_rate':[0.1,0.001,0.5,0.75,1], 'n_estimators':[100,500,1000], 'subsample':[0.1,0.5,1,1.5], 'min_samples_split':[2,4,6], 'max_depth':[None,5,10,20], 'random_state':[1], 'max_features':[None,3,7,10,15]}}
