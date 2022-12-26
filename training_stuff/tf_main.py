from preprocessing import convert_to_bag_of_words
from tensorflow.python.keras.models import load_model
from helper import predicted_label
import torch
import numpy as np
from helper import get_responses
from random import randint
from tensor_model import Model
query = ""
responses = get_responses()
TF_FILE = "./tf_checkpoint/model"
FILE = "./checkpoint/checkpoint.pth"
data = torch.load(FILE)
model = load_model(TF_FILE)
word_hash = data["word_hash"]
print("Model loaded successfully.")

PROB_THRESHOLD = 0.6 # if not above this threshold, we don't make a prediction.
while True:

    query = input("Type your query here: ")
    if query == 'q':
        break
    # then, convert query into bag_of_word then fit into our model
    X = convert_to_bag_of_words(word_hash, query)
    # X = torch.as_tensor(X)
    # if the word is not in our word list, we continue
    X = X.reshape(-1, 177)
    
    if sum(X[0]) == 0:
        print("AI: Sorry, I cannot understand you, please try another question")
        continue

    prediction = model.predict(X)
    
    probability = max(prediction[0])
    predicted = np.argmax(prediction[0])
    if probability < PROB_THRESHOLD:
        print("AI: Sorry, I cannot understand you, please try another question")
        continue
    # after we done getting our prediction, random a response 
    num_of_res = len(responses[predicted])
    print("AI: " + responses[predicted][randint(0, num_of_res - 1)])
    print("\t|Probability:", str(probability), "label predicted:", str(predicted) + "|")

    # print("Probability is: ", output)
