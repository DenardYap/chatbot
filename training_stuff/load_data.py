from preprocessing import get_data
import numpy as np
import torch
import pandas as pd 

train_data = pd.read_json("data.json")

# 'text' is the X, intent is the y
# Once classifed, random one response from 'responses'

bag_of_words, y, word_hash, word_count, Y_hash, sentence_count = get_data(train_data["intents"])

# Basically the class labels 
responses = [None for _ in range(len(Y_hash))]

for intent in Y_hash:
    responses[Y_hash[intent]] = intent

X = np.array(bag_of_words, dtype=np.float32)
y = np.array(y)