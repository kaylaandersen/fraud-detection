import cPickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import grid_search
from sklearn import metrics

# class skeleton
class Model(object):
    '''
    This class holds the skelton for common model instantiation, grid
    search, fitting and pickling procedures.
    Order of operations:
    1. Pass sklearn model class (not instanstiated), and parameter options for grid search to test.
    2. Run grid search on X and y training data. This will update the model hyperparameters used for the next functions to the best estimator from the gridsearch.
    3. Predict model. Returns y_pred
    4. Score Model.
    5. (Optional) For a model you want to reuse in the future, dump into a pickle file.
    '''

    def __init__(self, model_class, grid_search_dict):
        self.model = model_class()
        self.grid_search_dict = grid_search_dict

    def run_grid_search(self, X_train, y_train):
        # run grid search
        self.gs = grid_search.GridSearchCV(self.model,
                                           self.grid_search_dict).fit(X_train, y_train)
        # set self.model to the best estimator
        self.model = self.gs.best_estimator_

    def predict_model(self, X_test):
        self.y_pred = self.model.predict(X_test)
        return self.y_pred

    def score(self, y_true):
        return metrics.recall_score(y_true, self.y_pred)

    def dump_model(self, outpath):
        with open(outpath, 'w') as f:
            pickle.dump(self.model, f)

class ModelEvaluator(object):
    '''
    This class consolidates many of the metrics we may use to evaluate model performance.
    Input y_true and y_pred class labels
    '''

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def run_confusion_matrix(self, fig_outpath=None):
        self.confusion_matrix = metrics.confusion_matrix(y_true=self.y_true, y_pred=self.y_pred)
        if fig_outpath:
            fig, ax = plt.subplots(figsize=(2.5, 2.5))
            ax.matshow(self.confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(self.confusion_matrix.shape[0]):
                for j in range(self.confusion_matrix.shape[1]):
                    ax.text(x=j, y=1, s=self.confusion_matrix[i, j], va='center', ha='center')
            plt.xlabel('predicted label')
            plt.ylabel('true label')
            plt.savefig(fig_outpath)
