import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score,roc_curve, auc

##########################################################################
# My generel template for ML model testing

# TODO:
# 1) Raise Exception for not having the correct imports i.e. Scaler, imputer
#    These will be stored in the params arguement. TEST
# 4) Better unit testing and Exceptions - or any
# 5) MLTestTool needs training data and target column to run
#    parameter instances, and cv_folds are defined here as well. 
#    maybe a data handling class is needed for just data and what to do with it
#    MLTestTool could inherit the data from that parent class - open question
# 6) Clean the data and update the features?? 
# 7) Display full set of metrics for a given model: confusion matrix, 
#    class report, and stored metrics.
# 8) Generate ROC graph if auc is on?
##########################################################################

class MLTestTool:
    """
    A general ML test class that builds on sklearn.
    target varible and predictors must be flagged on input. 
    
    Attributes:
        training_df: The source of training data in pandas dataframe format
        target: The variable to be predicted
        cv_folds: integer, number of folds for cross-validation

    """

    def __init__(self, training_df, target, cv_folds=5, include_auc=False):
        
        self.df = training_df
        self.target = target
        self.cv_folds = cv_folds
        self.params = None
        self.include_auc = include_auc
        self.results = {}
        self.predictions = {}
        self.model_id = 0
        self.random_seed = 945945
        
# Helper Functions

    def _make_model_id(self):
        model_id = self.model_id 
        self.model_id += 1
        return model_id
        
    def _make_result(self, x):
            return {
                'mean': round(np.mean(x), 4),
                'sd': round(np.std(x), 4)
            }
    
# Evaluation Metrics

    def calculate_accuracies(self, y_hat, y_test): 
            true_positives = np.sum((y_hat == y_test)[y_hat == 1])
            true_negatives = np.sum((y_hat == y_test)[y_hat == 0])
            false_positives = np.sum((y_hat != y_test)[y_hat == 1])
            false_negatives = np.sum((y_hat != y_test)[y_hat == 0])

            recall = (
                true_positives
                / (true_positives + false_negatives)
            )

            precision = (
                true_positives
                / (true_positives + false_positives)
            )

            specificity = (
                true_negatives
                / (true_negatives + false_positives)
            )

            balanced_accuracy = (recall + specificity) / 2

            f1_score = 2 * (recall * precision) / (recall + precision)

            return recall, precision, specificity, balanced_accuracy, f1_score
    

    def test_model(self, params=None):
        """
            Function to perform the core machine learning analysis. 
            Metrics are calculated in Cross Validation and stored as a dictionary
            with average values and std. 

            Arguements:
                params: A dictionary of parameters, "model_instance" is required. 
                        Should be of the form:
                        params={'model_instance': <desired model>,
                                'scaler_instance': <optional scaler>,
                                'imputer_instance': <optional imputer>}
                

            Returns: A dictionary of tested models with corresponding metrics
        """

######################### Scale, CV, Imputation ################################   

        self.params = params
        
        if 'scaler_instance' in self.params.keys():
            raise Exception('No scaler defined in params.' \
                            'Use form {"scaler_instance":<scaler>}')
        if 'scaler_instance' in self.params:
            scaler = self.params['scaler_instance']
            scaled_x = scaler.fit_transform(X=self.df)
            X = scaled_x
            y = self.target.values
        else:
            X = self.df.values
            y= self.target.values
            
        accuracies = []
        balanced_accuracies = []
        recalls = []
        precisions = []
        specificities = []
        f1_scores = []
        auc = []

        y_hat_probs = []
        y_tests = []

        model_instance = self.params['model_instance']
        k_fold = KFold(n_splits=self.cv_folds, 
                       random_state=self.random_seed,
                       shuffle=True)
        
        if 'imputer_instance' in self.params.keys():
            raise Exception('No imputer defined in params.' \
                            'Use form {"imputer_instance":<imputer>}')
        if 'imputer_instance' in self.params:
            med_imp = self.params['imputer_instance']

        kf = k_fold.split(X, y)
        for train_index, test_index in kf:
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
        
            if 'imputer_instance' in self.params:
                X_train = med_imp.fit_transform(X_train)
                X_test = med_imp.fit_transform(X_test)

            trained_model = model_instance.fit(X=X_train, y=y_train)
            y_hat= trained_model.predict(X_test)
            y_hat_prob = [p[1] for p in trained_model.predict_proba(X_test)]
            accuracies.append(np.mean(y_hat == y_test)) 
            if self.include_auc:
                auc.append(roc_auc_score(y_test, y_hat_prob)) 

            recall, precision, specificity, balanced_accuracy, f1_score =\
                self.calculate_accuracies(y_hat, y_test)

            recalls.append(recall)
            precisions.append(precision)
            specificities.append(specificity)
            balanced_accuracies.append(balanced_accuracy)
            f1_scores.append(f1_score)

            y_hat_probs += y_hat_prob
            y_tests += y_test.tolist()
            
        
        model_id = self._make_model_id()
        if self.include_auc:
            self.results[model_id] = { 
                'model_id': model_id,
                'model': model_instance,
                'f1_score': self._make_result(f1_score),
                'recall': self._make_result(recalls),
                'precision': self._make_result(precisions),
                'specificity': self._make_result(specificities),
                'balanced_accuracy': self._make_result(balanced_accuracies),
                'accuracy': self._make_result(accuracies),
                'auc':self._make_result(auc)
            }
        else:
            self.results[model_id] = { 
                'model_id': model_id,
                'model': model_instance,
                'f1_score': self._make_result(f1_score),
                'recall': self._make_result(recalls),
                'precision': self._make_result(precisions),
                'specificity': self._make_result(specificities),
                'balanced_accuracy': self._make_result(balanced_accuracies),
                'accuracy': self._make_result(accuracies)
            }
            
        self.predictions[model_id] = {
            'prediction_probabilities': y_hat_probs,
            'y_test': y_tests,
        }
        return self.results

##################### Modeling ranking amd visualization #######################

    def rank_top_performers(self, metric='auc'):
        """
            Ranks models in the result dictionary by desired metric. 
            
            Arguements:
                metric:  one of; 'f1_score', 'recall', 'precision',
                        'specificity', 'balanced_accuracy', 'accuracy',
                        or 'auc'. 
            Returns: ordered list of the results dictionary
        """
        
        models = [(v['model_id'], v[metric]) for k, v in self.results.items()]
        models.sort(key=lambda x: x[1]['mean'], reverse=True)
        model_ids = [i[0] for i in models]

        return [self.results[m] for m in model_ids]

    def plot_features_importance(self, result):
        """
            Plots feature importance of models that retain feature weights. 
            most likely usable in tree boosted models. 
            
            Arguments: result; the trained model instance.
            Returns: Plot of ordered feature importance. 
        """
        
        if not hasattr(result, 'feature_importances_'):
            raise Exception('This model has no feature importances')
        feat_imp = pd.Series(
            result.feature_importances_,
            self.df.columns
        ).sort_values(ascending=False)
        plt.figure(figsize=(12,8))
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.tight_layout()
        
    def plot_ROC(self, model_id):
        if not self.include_auc:
            raise Exception("ROC AUC score has not been calculated." \
                           "Use include_auc=True and re-run test_model()")
            
        roc_model = self.results[model_id]['model']
        roc_train = self.test_model({'model_instance':roc_model})
        tested_models = len(self.results)-1
        
        lw = 2
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(
                self.predictions[tested_models]['y_test'],
                self.predictions[tested_models]['prediction_probabilities']
            )
            roc_auc[i] = auc(fpr[i], tpr[i])
        print(roc_model)
        plt.figure(figsize=(12,8))
        plt.plot(fpr[0], tpr[0], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()