import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

import statistics
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import MinMaxScaler

#from sklearn.metrics import roc_auc_score
##########################################################################
# My generel template for ML model testing


def test_model(df, model_instance, model_id, cv_folds=10, params=None):

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
######################### Over sampling technique ################################    
#     count_class_0, count_class_1 = df.anomalous.value_counts()
    
#     df_0 = df[df['anomalous'] == 0]
#     df_1 = df[df['anomalous'] == 1]
    
#     df_1_over = df_1.sample(count_class_0, replace=True, random_state=12347)
#     df = pd.concat([df_1_over, df_0], axis=0)

    scaler = MinMaxScaler()
    scaled_x = scaler.fit_transform(X=df.drop('event', axis=1))
    X = scaled_x
    y = df.event.values
    
    accuracies = []
    balanced_accuracies = []
#    aucs = []
    recalls = []
    precisions = []
    specificities = []
    f1_scores = []

    y_hat_probs = []
    y_tests = []
    
    
    folds = cv_folds
    k_fold = KFold(n_splits=folds, random_state=12347, shuffle=True)
#    med_imp = Imputer(missing_values='NaN',strategy='median',axis=0)

    kf = k_fold.split(X, y)
    for train_index, test_index in kf:
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        
#         X_train = med_imp.fit_transform(X_train)
#         X_test = med_imp.fit_transform(X_test)
        
        trained_model = model_instance.fit(X=X_train, y=y_train)
        y_hat= trained_model.predict(X_test)
        y_hat_prob = [p[1] for p in trained_model.predict_proba(X_test)]
        accuracies.append(np.mean(y_hat == y_test)) 
#        aucs.append(roc_auc_score(y_test, y_hat_prob, )) 

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
#        'auc': _make_result(aucs),
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
      
def rank_top_performers(model_dict, metric='auc'):

        models = [(v['model_id'], v[metric]) for k, v in model_dict.items()]
        models.sort(key=lambda x: x[1]['mean'], reverse=True)
        model_ids = [i[0] for i in models]

        return [model_dict[m] for m in model_ids]
    
def plot_features_importance(df, result):
    #model = result['model']
    feat_imp = pd.Series(
        result.feature_importances_,
        df.drop('event', axis=1).columns
    ).sort_values(ascending=False)
    plt.figure(figsize=(12,8))
    plt.tight_layout()
    feat_imp.plot(kind='bar', title='Feature Importances')
    