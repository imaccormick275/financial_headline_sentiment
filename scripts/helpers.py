# general
import pandas as pd
import numpy as np
import re

# nltk module
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import PorterStemmer 


def stem_sentence(sentence_as_list):
    """Function to stem each word within a sentence."""
    
    words = [PorterStemmer().stem(w) for w in sentence_as_list]

    return words


def remove_stop_words(sentence_as_list):
    """Function ro remove stop words from a sentence."""
    
    words = [w for w in sentence_as_list if w not in stopwords.words("english")] 
    
    return words


def split_sentence(sentence_as_string):
    """Function to split sentence into list of words."""
    
    # split all numbers (pos/neg int/float) from strings
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ? |[a-zA-Z$_&+,:;=?@#|<>.^*()%!-]+'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    sentence_as_string = list_to_string(rx.findall(sentence_as_string))
    
    # there are examples of where floating point numbers are separated with whitespace
    # e.g. eur0. 12m this converts to eur 0.12 m
    rx = re.compile(r'(\d+\s.\d+  |  [-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ? |[a-zA-Z$_&+,:;=?@#|<>.^*()%!-]+)', re.VERBOSE)

    words = rx.findall(sentence_as_string)

    for i, word in enumerate(words):
        words[i] = words[i].replace(" ", "")

    sentence_as_string = list_to_string(words)
    
    # word tokenize
    words = word_tokenize(sentence_as_string)
    
    return words


def list_to_comma_sep_string(sentence_as_list):
    """Function to convert a list into a comma separated string."""
    
    for i, word in enumerate(sentence_as_list):
        if i != (len(sentence_as_list)-1):
            sentence_as_list[i] = word+', '
    
    sentence = ''.join(sentence_as_list)
    
    return sentence


def list_to_string(sentence_as_list):
    """Function to convert a list into a space separated string."""
    
    for i, word in enumerate(sentence_as_list):
        if i!=len(sentence_as_list)-1:
            sentence_as_list[i] = word+' '
    
    sentence = ''.join(sentence_as_list)
    
    return sentence


def pos_tagging(sentence_as_list):
    """Function to tag sentences using NLTKs POS tagger."""
    
    words = nltk.pos_tag(sentence_as_list)
    
    return words


def getNodes(parent):
    """Function, helper function of get_regex."""
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == 'ROOT':
                pass
            else:
                if node.label() == 'NP JJ':
                    return node.leaves()
                
            getNodes(node)
    
    return []
            
            
def get_regex(sentence, grammar):
    """Function to find grammar pattern within text."""
    
    grammar = grammar.replace('∗','*')

    cp = nltk.RegexpParser(grammar) 

    result = cp.parse(sentence) 
       
    result = getNodes(result)
    
    sub = []
    for word_postag in result:
        word = word_postag[0]
        sub.append((word, word_postag[1]))
   
    return sub

    