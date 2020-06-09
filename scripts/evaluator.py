from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

def get_accuracy(y, y_hat):
    """Helper function to calculate accuracy"""
    return accuracy_score(y, y_hat)

def get_precision(y, y_hat):
    """Helper function to calculate precision"""
    return precision_score(y, y_hat, average='macro')

def get_recall(y, y_hat):
    """Helper function to calculate recall"""
    return recall_score(y, y_hat, average='macro')

def get_fscore(y, y_hat):
    """Helper function to calculate fscore"""
    return f1_score(y, y_hat, average='macro')
    

class evaluator:
    """Class to train, cross validate, test and generate performance metrics for specified model."""
    
    
    def __init__(self, experiment_name, model, X_train, y_train, X_test, y_test, k=5):
        """Method to initialise class.
        
        Args:
        model: model class which contains the following methods: model.fit, model.predict.
        X_train: training set.
        y_train: trainin set target variable.
        X_test: test set.
        y_test: test set target variable.
        k: number of folds for k fold cross validation.
        
        Return:
        None
        """
        
        # instatiate variables
        self.experiment_name = experiment_name
        self.y = y_test
        
        # cross validate and test model
        self.cross_val_metrics(model, X_train, y_train, k=k)
        self.test_metrics(model, X_train, y_train, X_test, y_test)       

    
    def cross_val_metrics(self, model, X, y, k=5):
        """Method to create cross validated metrics for specified model.
        
        Args:
        model: model class which contains the following methods: model.fit, model.predict.
        X: training set.
        y: training set target variable.
        k: number of folds for k fold cross validation.
        
        Return:
        None
        """
             
        accuracy_cv, precision_cv, recall_cv, fscore_cv = [], [], [], []
    
        skf = StratifiedKFold(n_splits=k)

        for train_index, test_index in skf.split(X, y):
                       
            X_train, X_val = X.iloc[train_index], X.iloc[test_index]
            y_train, y_val = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            
            if str(model.__class__) == "<class 'scripts.hierarchical_classifier.heirarchical_classifier'>":
                y_pred = model.predict(X_val, y_val)
            else:
                y_pred = model.predict(X_val)                
            
            accuracy_cv.append(get_accuracy(y_val, y_pred))
            precision_cv.append(get_precision(y_val, y_pred))
            recall_cv.append(get_recall(y_val, y_pred))
            fscore_cv.append(get_fscore(y_val, y_pred))
            
        self.accuracy_cv = np.array(accuracy_cv).mean()
        self.precision_cv = np.array(precision_cv).mean()
        self.recall_cv = np.array(recall_cv).mean()
        self.fscore_cv = np.array(fscore_cv).mean()  
        
    
    def test_metrics(self, model, X_train, y_train, X_test, y_test):
        """Method to create test metrics for specified model.
        
        Args:
        model: model class which contains the following methods: model.fit, model.predict.
        X_train: training set.
        y_train: trainin set target variable.
        X_test: test set.
        y_test: test set target variable
        
        Return:
        None
        """
        
        self.model = model
        
        self.model.fit(X_train, y_train)

        if str(model.__class__) == "<class 'scripts.hierarchical_classifier.heirarchical_classifier'>":
            self.y_pred = self.model.predict(X_test, y_test)
        else:
            self.y_pred = self.model.predict(X_test)


        self.accuracy_test = np.array(get_accuracy(y_test, self.y_pred)).mean()
        self.precision_test = np.array(get_precision(y_test, self.y_pred)).mean()
        self.recall_test = np.array(get_recall(y_test, self.y_pred)).mean()
        self.fscore_test = np.array(get_fscore(y_test, self.y_pred)).mean() 
       
    
    def get_results(self):
        """Method generate dataframe containing cross validated and test metrics."""
        
        # create dictionary for results (results_blank -- used to save dataframe to file)
        self.results_blank = {'experiment':[], 'result':[],'accuarcy':[], 
                              'precision':[], 'recall':[],  'fscore':[], 
                              'misclass_indicies':[], 'predictions':[]}
        results = self.results_blank

        # all results cross val
        results['experiment'].append(self.experiment_name)
        results['result'].append('cross validation')
        results['accuarcy'].append(round(self.accuracy_cv, 3))
        results['precision'].append(round(self.precision_cv, 3))
        results['recall'].append(round(self.recall_cv, 3))
        results['fscore'].append(round(self.fscore_cv, 3))
        results['misclass_indicies'].append(list())
        results['predictions'].append(list())
        
        # all results test
        results['experiment'].append(self.experiment_name)
        results['result'].append('test')
        results['accuarcy'].append(round(self.accuracy_test, 3))
        results['precision'].append(round(self.precision_test, 3))
        results['recall'].append(round(self.recall_test, 3))
        results['fscore'].append(round(self.fscore_test, 3))
        results['misclass_indicies'].append(list(self.y[(self.y != self.y_pred)].index))        
        results['predictions'].append(list(self.y_pred))   

        # create dataframe of results
        self.results = pd.DataFrame(results)

        return self.results
    
    
    def plot_confusion(self):
        """Method to plot confusion matrix for test."""
        # plot heatmap
        ax= plt.subplot()
        labels=list(np.unique(self.y))  
        cm = confusion_matrix(self.y, self.y_pred, labels=labels)
        sns.heatmap(cm, annot=True, ax = ax, fmt='d'); #annot=True to annotate cells

        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);
        
    
    def misclassified_examples(self, true_label=None, predicted_label=None):
        """Method to return all misclassified examples indecies."""
        if true_label==None and predicted_label==None:
            return self.y[(self.y != self.y_pred)].index

        return self.y[(self.y == true_label) & (self.y_pred == predicted_label)].index

       
    def save_results(self, save_path=''):
        """Method to save results to dataframe for comparison."""
        
        # load temp results
        if os.path.exists(save_path):
            all_results = pd.read_csv(save_path)
        else:
            all_results = pd.DataFrame(self.results_blank)
        
        # remove stored results if experiment has been run previously
        all_results = all_results[all_results['experiment']!=self.experiment_name]
        
        # update with current results
        all_results = pd.concat([all_results, self.results], sort=False)
    
        # save all results
        all_results.to_csv(save_path, index=None)