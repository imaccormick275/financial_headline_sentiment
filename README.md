### Udacity Machine Learning Engineer Nanodegree Program

# Capstone Project

## Financial Sentiment Analysis

### Introduction
With a wealth of information and opinion available in real time through news, social media and investor publications there is a vast source of financial textual data, which if understood correctly can be used by investors and traders to better understand companies and markets.

The aim of this project is to give retail investors access to statistics describing the sentiment of headlines for companies of interest, over a given historical time period. As such, the news articles will need be classified as positive, negative or neutral, from the point of view of a retail investor. 

This respository will focus on the development of a sentiment analysis algorithm to perform inference, as well as deploying the final algorithm so it can be integrated within a web application. The chosen model was deployed to an AWS Sagemaker endpoint. The source code for the web application, is avaliable in a seperate repository. 

### Included in this repository

#### Folder Structure
* datasets/ -- datasets on which models in notebooks directory are trained
* notebooks/ -- collection of notebooks documenting experimental analysis
* scripts/ -- all custom functions used in notebooks
* deployment/ -- deployment of final solution to endpoint to be accessed by webapp

#### Folder Contents
datasets
* FinancialDictionary -- folder containiny domain speicific lexicon
* FinancialPhraseBank -- financial phrase bank dataset
* processed_for_deployment -- data processed for S3 upload

notebooks
* 01 Process Datasets.ipynb -- download and process datasets
* 02 Feature Creation.ipynb -- all feature creation steps
* 03 Exploratory Analysis.ipynb -- exploritory analysis conducted
* 04 Experiment 0 Benchmark.ipynb -- benchmark model
* 05 Experiment 1 Association Rules.ipynb -- one of seven experiments conducted
* 06 Experiment 2 AssociationRules & HierarchicalClassification.ipynb -- one of seven experiments conducted
* 07 Experiment 3 CountVectorizedTags with GradientBoostingClassifier.ipynb -- one of seven experiments conducted
* 08 Experiment 4 CountVectorizedTags with GradientBoostingClassifier & HierarchicalClassifier.ipynb -- one of seven experiments conducted
* 09 Experiment 5 TFIDF with SupportVectorClassifier.ipynb -- one of seven experiments conducted
* 10 Experiment 6 TFIDF with SupportVectorClassifier & HierarchicalClassification.ipynb -- one of seven experiments conducted
* 11 Experiment 7 Ensemble & HierarchicalClassification.ipynb -- one of seven experiments conducted
* 12 Results.ipynb -- final results and model comparrison

deployment
* Deployment.ipynb -- deployment of final solution
* source_sklearn/ -- folder contianing model entry point scripts
        
scripts
* Test Scripts.ipynb -- notebook to run all test scripts
* apyori.py -- function to run apriori algorithm 
* test_helpers.py -- test scripts
* association_rules.py -- custom association ruls classifier
* config.py -- config containing global params used accross notebooks
* ensemble.py - custom ensemble classifier
* evaluator.py -- custom evaluation script
* helpers.py -- custom helper functions, predominantly used in association_rules.py and preprocessing notebook
* hierarchical_classifier.py -- custom hierarchical classifier   
        
### Dataset
* Financial phrase bank: https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10
* Domain specific dictionary: provided on request by Srikumar Krishnamoorthy the author of Sentiment Analysis of Financial News Articles using Performance Indicators, 2017.

### Environment
See requirements.txt for more details.

