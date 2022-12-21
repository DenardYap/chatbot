from train import model, word_hash
from preprocessing import convert_to_bag_of_words
from helper import predicted_label
import torch
import numpy as np
from helper import get_responses
from random import randint
query = ""
responses = get_responses()
while True:

    query = input("What can I do for you? (type 'q' to quit): ")
    if query == 'q':
        break
    # then, convert query into bag_of_word then fit into our model

    X = convert_to_bag_of_words(word_hash, query)
    X = torch.as_tensor(X)
    output = model(X)
    _, predicted = torch.max(output, dim = -1)
    
    # after we done getting our prediction, random a response 
    predicted = predicted.item()
    num_of_res = len(responses[predicted])
    print(responses[predicted][randint(0, num_of_res - 1)])
