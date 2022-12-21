# pre-process data including stemming/lemmatizing, and converting them into bag of words
import numpy as np
import pandas as pd
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
import torch

#nltk.download('punkt')

def extract(input_string):
    wordSet = set(string.punctuation).difference(set("'"))
    newS = ""
    # more efficient
    # we don't want any special characters, 
    # so get rid of them here
    for c in input_string:
        if c.isdigit() or c in wordSet:
            continue

        newS += c.lower()
            
    return newS.split()

def tokenize(list_of_word):
    
    # return nltk.word_tokenize(list_of_word)
    # not using nltk.word_tokenize() because I wnat to eliminate
    # filler words such as "the, a, it, he, etc" or numbers asap
    # this way we can boost the efficiency of the pre-processing
    return extract(list_of_word)

# consider doing lemming instead 
def stem(word):
    # stemmer = nltk.PorterStemmer()
    stemmer = SnowballStemmer(language="english")
    return stemmer.stem(word.lower())

# problem with lemmatization is we need pos = 'a' to change adjective
# such as 'better' to 'good', but we don't know if a word is adjective or not 
def lemm(word):
    
    lemmatizer = nltk.WordNetLemmatizer()
    return lemmatizer.lemmatize(word.lower())

# convert our text to a vector of words with 1 if that word is present in that sentence
def get_data(json_object : string):
    # first hash the word, then 

    word_count = 0
    Y_count = 0
    sentence_count = 0
    # first we need to find out how many unique word in the documnet
    # these text we care is in the 'text' section
    word_hash = {} # hash_table to store unique words
    Y_hash = {} # hash_table for the class
    stop_words = set(stopwords.words('english'))
    
    for obj in json_object:
        Y_hash[obj["intent"]] = Y_count
        Y_count += 1
        for sentence in obj["text"]:
            # split the sentence into list, so we can process easily
            cur_sentence = extract(sentence)
            
            # now hash the word(s) in cur_sentence
            for unique_word in cur_sentence:
                if unique_word in stop_words:
                    continue # don't want stop words to appear in our training data

                unique_word = stem(unique_word)
                if unique_word not in word_hash:
                    word_hash[unique_word] = word_count
                    word_count += 1
            sentence_count += 1
    print("Number of unique words found: " + str(word_count))

    # now we convert the hash table into a list/vector/np.array
    bag_of_words = [[0 for _ in range(word_count)] for _ in range(sentence_count)]
    y = [0 for _ in range(sentence_count)]
    # now we put a 1 at each respective position for where the word is presence 
    # for each sentence 
    cur_index = 0
    for obj in json_object:
        for sentence in obj["text"]:
            cur_sentence = extract(sentence)
            
            y[cur_index] = Y_hash[obj["intent"]]
            # now hash the word(s) in cur_sentence
            for unique_word in cur_sentence:
                if unique_word in stop_words:
                    continue # don't want stop words to appear in our training data
                unique_word = stem(unique_word)
                bag_of_words[cur_index][word_hash[unique_word]] = 1

            cur_index += 1

    return bag_of_words, y, word_hash, word_count, Y_hash, sentence_count

def convert_to_bag_of_words(word_hash, sentence):

    # Encode the str according to the word_hash
    
    cur_sentence = extract(sentence)
    
    stop_words = set(stopwords.words('english'))

    X = np.zeros(len(word_hash), dtype=np.float32)
    for unique_word in cur_sentence:
        if unique_word in stop_words or unique_word not in word_hash:
            continue # don't want stop words to appear in our training data
        unique_word = stem(unique_word)
        X[word_hash[unique_word]] = 1
    return X