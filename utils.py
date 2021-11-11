# Load the libraries
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
import re
# https://online.stat.psu.edu/stat504/lesson/1/1.7
import swifter
from sklearn.metrics import classification_report, accuracy_score


# Removing the html strips
def _strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# Removing the square brackets
def _remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# Define function for removing special characters
def _remove_special_characters(text):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


# Stemming the text
'''
Stemming (stem form reduction, normal form reduction) is the term used in information retrieval as well as in linguistic computer science to describe a procedure 
by which different morphological variants of a word are reduced to their common root, e.g. the declension of Wortes or words to Wort and conjugation of "gesehen" or "sah" to "seh". 
'''


def _simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


# removing the stopwords
def _remove_stopwords(text, is_lower_case=False):
    # Tokenization of text
    tokenizer = ToktokTokenizer()
    # Setting English stopwords
    stopword_list = nltk.corpus.stopwords.words('english')

    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token.lower() for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def preprocesser_text(df, to_prepro='review'):
    """
    The following text-normalization are applied:

    1.	removing the HTML tags. These are not relevant for understanding the input.
    2.	removing brackets with text (it seems that the text in brackets was either always a hyperlink or quote or reference, which are not too useful for the sentiment analysis).
    3.	removing special characters, e. g. ?,!,/
    4.	removing stop words. Stop words are common or “filler” words, which contain little information. For example, connection words. They are removed.
    5.	put words into the basic form with the Porter-Stemmer-Algorithm, which applies multiple, hardcoded rules to reduce the word-length. Some examples: `likes`, `liked`, `likely` and `liking` will all be reduced to `like`.
    """
    df[to_prepro] = df[to_prepro].swifter.apply(_strip_html)
    df[to_prepro] = df[to_prepro].swifter.apply(_remove_between_square_brackets)
    df[to_prepro] = df[to_prepro].swifter.apply(_remove_special_characters)
    df[to_prepro] = df[to_prepro].swifter.apply(_simple_stemmer)
    df[to_prepro] = df[to_prepro].swifter.apply(_remove_stopwords)
    return df


def preprocesser_bert(df, to_prepro='review'):
    """       
     The following text-normalization are applied:

    1.	removing the HTML tags. These are not relevant for understanding the input.
    2.	removing brackets with text (it seems that the text in brackets was either always a hyperlink or quote or reference, which are not too useful for the sentiment analysis).
    3.	removing special characters, e. g. ?,!,/
    """

    df[to_prepro] = df[to_prepro].swifter.apply(_strip_html)
    df[to_prepro] = df[to_prepro].swifter.apply(_remove_between_square_brackets)
    df[to_prepro] = df[to_prepro].swifter.apply(_remove_special_characters)
    # df[to_prepro]=df[to_prepro].swifter.apply(_simple_stemmer)
    # df[to_prepro]=df[to_prepro].swifter.apply(_remove_stopwords)
    return df


def binarize_sentiment(series, dict_={'positive': 1, 'negative': 0}):
    return series.replace(to_replace=dict_)


def train_test_split(df, train_n=40000):
    test = df.iloc[train_n:]
    train = df.iloc[:train_n]
    return train, test


def evaluate(y_true, y_pred,
             target_names=['Negative', 'Positive']):  # Target names are probably right? Could be wrong though.
    report = classification_report(y_true, y_pred, target_names=target_names)
    return accuracy_score(y_true, y_pred), report
