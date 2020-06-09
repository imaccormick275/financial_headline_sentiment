import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from scripts.helpers import get_regex
from scripts.helpers import split_sentence
from scripts.helpers import list_to_comma_sep_string
from scripts.helpers import list_to_string
from scripts.helpers import pos_tagging

from scripts.apyori import apriori


def append_to_list(tags, sentiment):
    """Helper function used to append tag to sentiment to be applied row-wise in dataframe using apply method."""
    
    tags_list = tags.split(', ')
    tags_list.append(sentiment)

    return tags_list

def get_transactions(X, y):
    """Helper function used to create transactions for apriori."""
    
    df_temp = pd.merge(X,y,left_index=True, right_index=True)
    df_temp.rename(columns = {df_temp.columns[0]: 'tags'}, inplace = True)
    df_temp.rename(columns = {df_temp.columns[1]: 'sentiment'}, inplace = True)
    transactions = list(df_temp.apply(lambda x: append_to_list(x.tags, x.sentiment), axis=1))
    return transactions


class association_classifier:
    """Class to make class predictions using association rules."""
    
    def __init__(self, min_support, min_confidence, min_lift):
        """Method to initialise class.
        
        Args
        min_suuport: float
        min_confidence: float
        min_lift: float
        
        Return:
        None
        """
        
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
    
    
    def __copy__(self):
        """Method to copy object instance."""

        return association_classifier(self.min_support, self.min_confidence, self.min_lift)

    
    def fit(self, X_train, y_train):
        """Method to train the classifier using input data
        
        Args:
        X_train: pandas dataframe or series. train set.
        y_train: pandas series. train set target varibale. 
        
        Return:
        None
        """
        
        # get transactions for apriori algorithm
        transactions = get_transactions(X_train, y_train)
       
        # get itemset
        itemset = apriori(transactions, min_support=self.min_support, 
                        min_confidence=self.min_confidence, min_lift=self.min_lift)
        itemset = list(itemset)

        # empty dicts to store results
        association_rules = {'antecedents':[],'consequent_sentiment':[],'confidence':[], 
                             'support':[], 'antecedent_length':[]}
        sentiments = set(y_train.unique())

        # iterate over rules and covert to dataframe extracting required rule info to make predictions
        for i, item in enumerate(itemset):
            confidence = list(list(itemset[i])[2][0])[2]
            support = list(itemset[i])[1]
            consequent_sentiment = list(itemset[i][2][0][1])
            antecedents = list(itemset[i][2][0][0])
            antecedent_length = len(list(antecedents))

            # if rule contains exactly one sentiment to be predicted record rule ()
            if len(sentiments.intersection(set(consequent_sentiment)))==1 and len(list(consequent_sentiment))==1:
                association_rules['antecedents'].append(list_to_comma_sep_string(antecedents))
                association_rules['consequent_sentiment'].append(list_to_comma_sep_string(consequent_sentiment))
                association_rules['confidence'].append(confidence)
                association_rules['support'].append(support)
                association_rules['antecedent_length'].append(antecedent_length)

        # convert to dataframe
        self.df_association_rules = pd.DataFrame(association_rules)
        self.df_association_rules.sort_values('confidence', ascending=False, inplace=True)
        self.df_association_rules.reset_index(drop=True, inplace=True)
        
        # empty tag has to appear in association rules
        if '' not in list(self.df_association_rules['antecedents']) and 'neutral' in sentiments:
            empty = pd.DataFrame({'antecedents':[''],'consequent_sentiment':['neutral'],'confidence':[1],'support':[1],'antecedent_length':[1]})
            self.df_association_rules = pd.concat([self.df_association_rules, empty])

    def predict(self, X_test):
        """Method to make predictions, vectorized implementation
        
        Args
        X_test: pandas dataframe or dataseries. test set.
        
        Return:
        y_pred: pandas dataseries. predictions of model.
        """
        
        ## exact match
        unique_classes = np.unique(self.df_association_rules['consequent_sentiment'].to_numpy()).reshape(1,-1)
                         
        # params for array dimensions
        num_rules = self.df_association_rules.shape[0]
        num_examples = X_test.shape[0]
        num_classes = unique_classes.shape[1]


        E = np.repeat(self.df_association_rules['antecedents'].to_numpy().reshape(1,-1), num_examples, axis=0)
        R = np.repeat(X_test.to_numpy().reshape(-1,1), num_rules, axis=1)

        A = np.repeat(self.df_association_rules['consequent_sentiment'].to_numpy().reshape(-1,1), num_classes, axis=1)
        B = np.repeat(unique_classes, num_rules, axis=0)
        C = self.df_association_rules['confidence'].to_numpy().reshape(-1,1)

        exact_match_confidence = np.matmul(E==R, np.multiply((A==B), C))

        ## partial match
        my_vocabulary = list(self.df_association_rules[self.df_association_rules['antecedents'].apply(lambda x: len(x.split()))==1]['antecedents'])
        my_vocabulary_dict = {}
        for i, vocab in enumerate(my_vocabulary):
            my_vocabulary_dict[vocab] = i    

        vectorizer = CountVectorizer(lowercase = False, token_pattern = '[a-zA-Z0-9$&+,:;=?@#|<>.^*()%!-]+')
        vectorizer.fit_transform(my_vocabulary_dict)
        vectorizer.vocabulary_ = my_vocabulary_dict
        tf = vectorizer.transform(X_test.apply(lambda s: s.replace(',', '')))
        tf = tf.todense()

        df_unique_ass_rules = self.df_association_rules[self.df_association_rules['antecedents'].apply(lambda x: len(x.split()))==1]
        num_rules = df_unique_ass_rules.shape[0]

        A = np.repeat(df_unique_ass_rules['consequent_sentiment'].to_numpy().reshape(-1,1), num_classes, axis=1)
        B = np.repeat(unique_classes, num_rules, axis=0)
        C = df_unique_ass_rules['confidence'].to_numpy().reshape(-1,1)

        conf_sums = np.matmul(tf, np.multiply((A==B), C))
        conf_counts = np.matmul(tf, np.multiply((A==B), C)!=0)
        conf_counts = ((conf_counts==0)*0.000001)+conf_counts

        partial_match_confidence = np.divide(conf_sums, conf_counts)

        ## if exact match: exact_match_confidence else partial_match_confidence
        if_not_exact_match = (exact_match_confidence.sum(axis=1)==0).reshape(-1,1)
        confidence = exact_match_confidence + np.multiply(if_not_exact_match, partial_match_confidence)
        
        ## if there are no exact or partial matches predict the most common class
        classes = np.repeat(unique_classes, num_examples, axis=0)
        if 'neutral' in list(classes[0]):
            i = list(classes[0]).index('neutral')
            confidence[:,i:i+1] = confidence[:,i:i+1] + (confidence.sum(axis=1)==0)*0.6
        elif 'positive' in list(classes[0]):
            i = list(classes[0]).index('positive')
            confidence[:,i:i+1] = confidence[:,i:i+1] + (confidence.sum(axis=1)==0)*0.3   

        ## get predictions
        classes = np.repeat(unique_classes, num_examples, axis=0)
        y_pred = classes[np.unravel_index(confidence.argmax(axis=1), classes.shape)]
        
        # confidence for predict_proba
        self.confidence = confidence
        
        y_pred = y_pred.reshape(-1,)
        
        # return prediction
        return y_pred
    
    def predict_proba(self, X_test):
        return np.array(self.confidence)
