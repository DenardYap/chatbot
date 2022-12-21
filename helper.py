import torch
import numpy as np
import pandas as pd 
# get the predicted label
def predicted_label(logits):
    """Determine predicted class index given logits.

    Returns:
        the predicted class output as a PyTorch Tensor
    """
    pred = []
    for t in logits:
        current_max = float("-infinity")
        max_i = 0
        for i in range(len(t)):
            if t[i] > current_max:
                current_max = t[i]
                max_i = i   

        pred.append(max_i)
    return torch.as_tensor(pred)

def get_responses():

    # a hash table where key is intent, and value is list of reponses
    intents_and_responses = {}
    
    data = pd.read_json("data.json")
    for i, obj in enumerate(data["intents"]):
        intents_and_responses[i] = obj["responses"]
    return intents_and_responses