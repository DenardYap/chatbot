from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
import tensorflowjs as tfjs

class Model():
    
    def __init__(self, input_size, output_size):
        self.hidden_size = 8
        self.model = Sequential()
        self.model.add(Dense(units=self.hidden_size, activation="relu", input_dim=input_size))
        self.model.add(Dense(units=self.hidden_size, activation="relu"))
        self.model.add(Dense(units=output_size, activation="softmax"))
        # maybe use F1 instead
        self.model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy")

    def fit(self, X_train, y_train, batch_size, epochs):
        self.model.fit(X_train, y_train, batch_size, epochs)

    def save(self, file_name):
        self.model.save(file_name)

    def savejs(self, file_name):
        
        tfjs.converters.save_keras_model(self.model, file_name)