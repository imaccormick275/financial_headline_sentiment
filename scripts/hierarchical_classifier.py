import pandas as pd
import numpy as np

class heirarchical_classifier:
    """Class to create heirarchial classification model from two sub models. Built specifically for the three class case
    where two classes are to be combined into one class."""
    
    def __init__(self, model_h1, model_h2, y_test, classes_from, class_to, class_remain):
        """Function to initialise class.
        
        Args:
        model_h1: model class which contains the following methods: model.fit, model.predict.
        model_h2: model class which contains the following methods: model.fit, model.predict.
        y_test: pandas Series. 
        classes_from: list.
        class_to: string.
        class_remain: string. 
        
        Return:
        None
        """
    
        self.model_h1 = model_h1
        self.model_h2 = model_h2
        self.y_test = y_test.copy()
        
        self.classes_from = classes_from
        self.class_to = class_to
        self.class_remain = class_remain
    
    def fit(self, X_train, y_train):
        """Function to fit both models.
        
        Args:
        X_train: pandas dataframe. Train set.
        y_train: pandas series. Train set target variable. 
        
        Return:
        None
        """
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        
        self.X_trian_h1 = self.X_train.filter(list(self.y_train[(self.y_train != self.class_remain)].index), axis=0)
        self.y_train_h1 = self.y_train.filter(list(self.y_train[(self.y_train != self.class_remain)].index), axis=0)
        self.X_trian_h2 = self.X_train
        self.y_train_h2 = self.y_train.replace(self.classes_from, self.class_to)
               
        # train model 1
        self.model_h1.fit(self.X_trian_h1, self.y_train_h1)
        
        # train model 2
        self.model_h2.fit(self.X_trian_h2, self.y_train_h2)
        
    def predict(self, X_test, y_test):
        """Function generate predictions using both models.
        
        Args:
        X_test: pandas dataframe. Test set.
        y_test: pandas series. Test set target variable.
        
        Return:
        y_pred: pandas dataseries. predictions of model.
        """
             
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        
        
        self.X_test_h1 = self.X_test.filter(list(self.y_test[(self.y_test != self.class_remain)].index), axis=0)
        self.y_test_h1 = self.y_test.filter(list(self.y_test[(self.y_test != self.class_remain)].index), axis=0)
        self.X_test_h2 = self.X_test #
        self.y_test_h2 = self.y_test.replace(self.classes_from, self.class_to)  
        
        
        self.y_pred_h1 = self.model_h1.predict(self.X_test_h1)
        self.y_pred_h1 = pd.Series(self.y_pred_h1, self.X_test_h1.index)
        
        self.y_pred_h2 = self.model_h2.predict(self.X_test_h2)
        self.y_pred_h2 = pd.Series(self.y_pred_h2, self.X_test_h2.index)
        
        
        self.X_test_h2_sub = self.X_test_h2[self.y_pred_h2==self.class_to]
        self.y_pred_h1_sub = self.model_h1.predict(self.X_test_h2_sub)
        self.y_pred_h1_sub = pd.Series(self.y_pred_h1_sub, self.X_test_h2_sub.index)
        
        self.y_pred = self.y_pred_h1_sub.combine_first(self.y_pred_h2).sort_index()
        
        return self.y_pred
    
    def predict_proba(self, X_test):
        """Function to ..."""
        probs_1 = self.model_h1.predict_proba(self.X_test_h2_sub)
        probs_2 = self.model_h2.predict_proba(self.X_test_h2)
        
        classes_1 = self.classes_from
        classes_2 = [self.class_to, self.class_remain]
        
        indicies_1 = self.X_test_h2_sub.index
        indicies_2 = self.X_test_h2.index
        

        neutr_probs = probs_2[:,0:1]

        neu_probs = pd.DataFrame(neutr_probs, index=indicies_2, columns=[self.class_remain])
        pol_probs = pd.DataFrame(probs_1, index=indicies_1, columns=classes_1)
        
       
        probs = pd.merge(neu_probs, pol_probs, 'left', left_index=True, right_index=True)
        probs = probs.fillna(0.0)
        probs = probs.to_numpy()
        
        return probs
        
    
    def get_sub(self):
        """Function to return predictions and actuals from sub models."""

        return self.y_pred_h1, self.y_test_h1, self.y_pred_h2, self.y_test_h2
        