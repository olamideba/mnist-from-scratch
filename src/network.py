from typing import Any


import numpy as np
import random


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # the (y, 1) is the shape of the vector for each respective hidden layer and output for the bias
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] # This represents the connections between Layer 1 → Layer 2, and Layer 2 → Layer 3.
        
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
     
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))