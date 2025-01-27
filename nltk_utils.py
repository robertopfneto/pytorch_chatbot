import nltk
nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer # for stemming

stemmer = PorterStemmer()

#Pre-processing techniques 

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def setm(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentencec, all_words):
    pass

