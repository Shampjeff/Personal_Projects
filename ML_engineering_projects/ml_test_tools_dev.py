import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

import statistics
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
#cross_val_predict from model_selection in sklearn
##########################################################################
# My generel template for ML model testing

# TODO:
# 1) Raise Exception for not having the correct imports i.e. Scaler, imputer
#    These will be stored in the params arguement. 
# 2) Auto index tested models. Required as input currently
# 3) Add ROC SUC score and make it an on/off feature. Maybe a kwarg?
# 4) Better unit testing and Exceptions - or any
# 5) MLTestTool needs training data and target column to run
#    parameter instances, and cv_folds are defined here as well. 
#    maybe a data handling class is needed for just data and what to do with it
#    MLTestTool could inherit the data from that parent class

class MLTestTool:
    """
    A general ML test class that builds on sklearn.
    target varible and predictors must be flagged on input. 
    
    Attributes:
        dataframe: the source of data in pandas dataframe format
        target: The variable to be predicted
        cv_folds: integer, number of folds for cross-validation
        params: dictionary of instances; scaler, imputer, etc that 
            maybe be need depending on the task. 
            
    Key Word Arguements:
        impute: toogles an imputer instance in params
        scale: toggles a scaler instance in params

    """

    def __init__(self, training_df, target, cv_folds=5, params=None, **kwargs):
        
        self.df = training_df
        self.target = target
        self.cv_folds = cv_folds
        self.params = params
        self.random_seed = 945945
    
# Evaluation Metrics
    def test_model(self, model_instance, model_id, **kwargs):
        """
            Function to perform the core machine learning analysis. 
            Metrics are calculated in Cross Validation and stored as a dictionary
            with average values and std. 

            Arguement:
                model_instance: The parameterized model you would like to test
                model_id: An interger sequence to index the models tested

            Returns: A dictionary of tested models with corresponding metrics
        """
    
        predictions = {}
        results = {}
    
        def calculate_accuracies(y_hat, y_test): 
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
    

        def _make_result(x):
            return {
                'mean': round(np.mean(x), 4),
                'sd': round(np.std(x), 4)
            }
######################### Scale, CV, Imputation ################################   

        if 'scale' in kwargs:
            scaler = self.params['scaler_instance']
            scaled_x = scaler.fit_transform(X=df.drop(self.target, axis=1))
            X = scaled_x
            y = df.event.values
        else:
            X = self.df.values
            y= self.target.values
            
        accuracies = []
        balanced_accuracies = []
        recalls = []
        precisions = []
        specificities = []
        f1_scores = []

        y_hat_probs = []
        y_tests = []


        k_fold = KFold(n_splits=self.cv_folds, 
                       random_state=self.random_seed,
                       shuffle=True)
        
        if 'impute' in kwargs:
            med_imp = self.params['imputer_instance']

        kf = k_fold.split(X, y)
        for train_index, test_index in kf:
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
        
            if 'impute' in kwargs:
                X_train = med_imp.fit_transform(X_train)
                X_test = med_imp.fit_transform(X_test)

            trained_model = model_instance.fit(X=X_train, y=y_train)
            y_hat= trained_model.predict(X_test)
            y_hat_prob = [p[1] for p in trained_model.predict_proba(X_test)]
            accuracies.append(np.mean(y_hat == y_test)) 

            recall, precision, specificity, balanced_accuracy, f1_score =\
                calculate_accuracies(y_hat, y_test)

            recalls.append(recall)
            precisions.append(precision)
            specificities.append(specificity)
            balanced_accuracies.append(balanced_accuracy)
            f1_scores.append(f1_score)

            y_hat_probs += y_hat_prob
            y_tests += y_test.tolist()

        results = { 
            'model_id': model_id,
            'model': model_instance,
            'f1_score': _make_result(f1_score),
            'recall': _make_result(recalls),
            'precision': _make_result(precisions),
            'specificity': _make_result(specificities),
            'balanced_accuracy': _make_result(balanced_accuracies),
            'accuracy': _make_result(accuracies)
        }

        predictions[model_id] = {
            'prediction_probabilities': y_hat_probs,
            'y_test': y_tests,
        }
        return results

# Modeling ranking

    def rank_top_performers(self,model_dict, metric='auc'):

            models = [(v['model_id'], v[metric]) for k, v in model_dict.items()]
            models.sort(key=lambda x: x[1]['mean'], reverse=True)
            model_ids = [i[0] for i in models]

            return [model_dict[m] for m in model_ids]

# Feature Importance

    def plot_features_importance(self, result):
        feat_imp = pd.Series(
            result.feature_importances_,
            self.df.drop(self.target, axis=1).columns
        ).sort_values(ascending=False)
        plt.figure(figsize=(12,8))
        plt.tight_layout()
        feat_imp.plot(kind='bar', title='Feature Importances')
    