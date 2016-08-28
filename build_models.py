import cPickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import grid_search, metrics

# class skeleton
class Model(object):
    """
    This class holds the skelton for common model implementations.

    Inlcudes ability to grid search, fitting and pickling procedures.
    Suggested order of operations:
    1. Pass sklearn model class (not instanstiated)
    2. Run grid search on X and y training data. This will update the model
       hyperparameters used for the next functions to the best estimator from
       the gridsearch.
    3. Predict model. Returns y_pred.
    4. Score Mmdel.
    5. (Optional) For a model you want to reuse, dump into a pickle file.
    """

    def __init__(self, model_class, **kwargs):
        """Instantiates an sklearn model, with optional parameters.

        Args:
        model_class (class): A sklearn model class, not instanstiated.
        **kwargs: Parameters for the sklearmn model class, if desired.

        """
        self.model = model_class(**kwargs) # instatiate model with params

    def run_grid_search(self, grid_search_dict, X_train, y_train):
        """Runs grid search and sets the self.model to the best estimator.

        Args:
        grid_search_dict (dict): A nested dictionary of hyperparameters to test.
        X_train (array): Training dataset features (2d).
        y_train (array): Training dataset labels (1d).

        """
        # run grid search
        self.gs = grid_search.GridSearchCV(self.model, grid_search_dict)
        # fit gs to training data
        self.gs.fit(X_train, y_train)
        # set self.model to the best estimator
        self.model = self.gs.best_estimator_

    def predict_model(self, X_test):
        """Predicts labels from test dataset labels.

        Args:
        X_test (array): Test dataset features (2d).

        """
        self.y_pred = self.model.predict(X_test)
        return self.y_pred

    def score(self, y_true, metric):
        """Scores the current prediction compared to the true values.

        Args:
        y_true (array): Test dataset true values (1d).
        metric (function): Sklearn model function defintion to run.

        """
        return metric(y_true, self.y_pred)

    def dump_model(self, outpath):
        """Saves the current model definition into a pickle file.

        Args:
        outpath (string): Path to save the pickle file to.

        """
        with open(outpath, 'w') as f:
            pickle.dump(self.model, f)

class ModelEvaluator(object):
    '''
    NOTE THIS IS NOT FULLY IMPLEMENTED YET
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
