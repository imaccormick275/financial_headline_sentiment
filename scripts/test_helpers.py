from helpers import getNodes
from helpers import get_regex
from helpers import stem_sentence
from helpers import remove_stop_words
from helpers import split_sentence
from helpers import list_to_comma_sep_string
from helpers import list_to_string
from helpers import pos_tagging

# split_sentence tests
def test_split_sentence_01():
    """Function to test 'split_sentence' - empty split."""
    assert(split_sentence('') == [])

def test_split_sentence_02():    
    """Function to test 'split_sentence' - basic split."""
    assert(split_sentence('the rain in spain') == ['the', 'rain', 'in', 'spain'])

def test_split_sentence_03():     
    """Function to test 'split_sentence' -  basic split with symbots."""
    assert(split_sentence('!the rain in spain?') == ['!','the', 'rain', 'in', 'spain','?'])

def test_split_sentence_04():
    """Function to test 'split_sentence' - numerical splits variation 1."""
    assert(split_sentence('eur50m') == ['eur', '50', 'm'])

def test_split_sentence_05():
    """Function to test 'split_sentence' - numerical splits variation 2.""" 
    assert(split_sentence('eur50.0m') == ['eur', '50.0', 'm'])
    
def test_split_sentence_06():
    """Function to test 'split_sentence' - numerical splits variation 3."""
    assert(split_sentence('-50m') == ['-50', 'm'])

def test_split_sentence_07():
    """Function to test 'split_sentence' - does not split words separated by underscores."""
    assert(split_sentence('operating_profit and') == ['operating_profit', 'and'])

    
# list_to_comma_sep_string tests
def test_list_to_comma_sep_string_01():
    """Function to test 'list_to_comma_sep_string' - empty list."""
    # empty list
    assert(list_to_comma_sep_string([]) == '')

def test_list_to_comma_sep_string_02():
    """Function to test 'list_to_comma_sep_string' - one word in list."""
    assert(list_to_comma_sep_string(['abc']) == 'abc')

def test_list_to_comma_sep_string_03():
    """Function to test 'list_to_comma_sep_string' - multi word list."""
    assert(list_to_comma_sep_string(['abc', 'def']) == 'abc, def')

    
# list_to_string tests
def test_list_to_string_01():
    """Function to test 'list_to_comma_sep_string' - empty list."""
    assert(list_to_string([]) == '')
    
def test_list_to_string_02():
    """Function to test 'list_to_comma_sep_string' - one word in list."""
    assert(list_to_string(['abc']) == 'abc')

def test_list_to_string_03():
    """Function to test 'list_to_comma_sep_string' - multi word list."""
    assert(list_to_string(['abc', 'def']) == 'abc def')

    
# pos_tagging tests
def test_pos_tagging_01():
    """Function to test 'pos_tagging' - empty list."""
    assert(pos_tagging([]) == [])

def test_pos_tagging_02():
    """Function to test 'pos_tagging' - example."""
    assert(pos_tagging(['the', 'rain', 'in']) == [('the', 'DT'), ('rain', 'NN'), ('in', 'IN')])