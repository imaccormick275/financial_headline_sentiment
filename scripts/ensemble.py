import pandas as pd
import numpy as np

def reorder_probs(probs, preds, class_order):
    '''Helper function to reorder of sklearn class probability outputs to match association classifier class order.'''
    
    # get index mapping (refactor do not need to iterate over all of test predictions)
    prob_pred_index = probs.argmax(axis=1)
    mapping={}
    for i, pred in enumerate(list(preds)):
        mapping[pred]=prob_pred_index[i]
    
    # reorder permutation
    permutation = []
    for i, class_ in enumerate(class_order):
        permutation.append(mapping[class_])

    
    # reorder probability matrix
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))
    probs = probs[:, idx]      
    
    return probs

class ensemble:
    """Ensemble methods to combine several machine learning techniques into one predictive model."""
    def __init__(self, models_feats, y_test):
        # init features
        self.models_feats = models_feats
        self.y_test = y_test
        
    def fit(self, X_train, y_train):
        """ Method to fit enemble.
        
        Args:
        X_train: pandas dataframe. Train set.
        y_train: pandas series. Train set target variable. 
        
        Return:
        None
        """
        # order of classes
        self.class_order = list(np.unique(y_train))
        
        # train all models
        for i, model in enumerate(list(self.models_feats.keys())):
            X_train_model = X_train[self.models_feats[model]]
            model.fit(X_train_model, y_train)
        
    def predict(self, X_test):
        """ Method to predict using pre-fitted ensemble.

        Args:
        X_test: pandas dataframe. Test set.
        
        Return:
        None
        """
        
        # empty list to store probabilities of classes for predictions
        model_probs = []
        
        # get predicted porbabilites of each class for each model
        for i, model in enumerate(list(self.models_feats.keys())):
            X_test_model = X_test[self.models_feats[model]] 
            
            if str(model.__class__) == "<class 'scripts.hierarchical_classifier.heirarchical_classifier'>":
                preds = model.predict(X_test_model, self.y_test)
            else:
                preds = model.predict(X_test_model)
                
            probs = model.predict_proba(X_test_model)
            probs = reorder_probs(probs, preds, self.class_order)
            model_probs.append(probs)
        
        # take max probability accross all models
        total_probs = np.zeros_like(probs)
        for i in range(len(model_probs)):
            total_probs = np.maximum(total_probs, model_probs[i])
        
        # get predictions
        classes = np.repeat(np.array(self.class_order).reshape(1,-1), X_test.shape[0], axis=0)        
        y_pred = classes[np.unravel_index(total_probs.argmax(axis=1), classes.shape)]    
        
        return y_pred
 
            