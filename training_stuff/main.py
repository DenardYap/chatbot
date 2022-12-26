from preprocessing import convert_to_bag_of_words
from helper import predicted_label
import torch
import numpy as np
from helper import get_responses
from random import randint
from model import Model
query = ""
responses = get_responses()

FILE = "./checkpoint/checkpoint.pth"
FILE2 = "./checkpoint/1000_checkpoint.pth"
data = torch.load(FILE)
model_data = torch.load(FILE2)
input_size = data["input_size"]
word_hash = data["word_hash"]
output_size = data["output_size"]
model_state = model_data["model_state"]
model = Model(input_size, output_size)
model.load_state_dict(model_state)
model.eval()
print("Model loaded successfully.")

PROB_THRESHOLD = 0.5 # if not above this threshold, we don't make a prediction.
while True:

    query = input("Type your query here: ")
    if query == 'q':
        break
    # then, convert query into bag_of_word then fit into our model
    with torch.no_grad():
        X = convert_to_bag_of_words(word_hash, query)
        X = torch.as_tensor(X)

        # if the word is not in our word list, we continue
        if sum(X) == 0:
            print("AI: Sorry, I cannot understand you, please try another question")
            continue

        output = model(X)
        print(output)
        probability, predicted = torch.max(output, dim = -1)
        print("Probability: ", probability)
        print("Predicted: ", predicted)
        if probability < PROB_THRESHOLD:
            print("AI: Sorry, I cannot understand you, please try another question")
            continue
        # after we done getting our prediction, random a response 
        predicted = predicted.item()
        num_of_res = len(responses[predicted])
        print("AI: " + responses[predicted][randint(0, num_of_res - 1)])
        # print("Probability is: ", output)
