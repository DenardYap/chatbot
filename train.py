from load_data import X, y, responses, word_hash, word_count, sentence_count
import sklearn 
from preprocessing import convert_to_bag_of_words
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from model import Model
from helper import predicted_label

# shuffle has to be on since y is sequential
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=69
# )

class ChatBot(Dataset):

    def __init__(self):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples 

lr = 0.001
batch_size = 4
num_epoch = 1000
dataset = ChatBot()
train_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
input_size = word_count # number of unique words
output_size = len(responses) # number of intents
model = Model(input_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(num_epoch):

    correct = 0
    for (X, y) in train_loader: 
        # reset the gradients
        optimizer.zero_grad()
        # feed forward our input (feed forward)
        outputs = model(X)

        # calculate the loss 
        loss = criterion(outputs, y)
        
        # update the gradient (backprop) 
        loss.backward()

        # next step
        optimizer.step()
        if (epoch+1) % 50 == 0:
            predictions = predicted_label(outputs.data)
            for i in range(len(predictions)):
                correct += (predictions[i] == y[i])

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epoch} : loss = {loss.item():.5f} | Accuracy = {correct/sentence_count}")


# print(f"Final training loss is: {loss.item():.5f}", )