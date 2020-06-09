from __future__ import print_function

import argparse
import os
import pandas as pd
import json

from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        request_decoded = request_body.decode("UTF-8")
        json_format = json.loads(request_decoded)
        request_data = json_format['data']
        
        return request_data
    
    if request_content_type == 'text/csv':
        #request_decoded = request_body.decode("UTF-8")
        json_format = json.loads(request_body)
        request_data = json_format['data']
        
        return request_data
        
    else:
        raise ValueError("{} request content type is not supported by this script.".format(request_content_type))


def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    return {'predictions': list(predictions)}


def output_fn(prediction, accept):
    if accept == "application/json":
        json_format = json.dumps(prediction)
        prediction_encoded = json_format.encode('UTF-8')
        return prediction_encoded
    
    elif accept == 'text/csv':
        json_format = json.dumps(prediction)
        prediction_encoded = json_format.encode('UTF-8')
        return prediction_encoded

    else:
        raise ValueError("{} accept type is not supported by this script.".format(accept))
        
        

if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
        
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "dp1_train.csv"))

    # Labels are in the first column
    train_y = train_data.sentiment
    train_x = train_data.text
    
    # vocabulary - words in first column
    vocab = pd.read_csv(os.path.join(training_dir, "dp1_vocab.csv"))
    vocab = list(vocab.word)
    
    ## Define model 
    model = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab, ngram_range=(1,2))),
                      ('model', LinearSVC())])         
    
    ## Train model
    model.fit(train_x,train_y)   

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))