import numpy as np
from sklearn.datasets import fetch_mldata
import random
saved_mnist = '/home/rubab/PycharmProjects/neural_network'
mnist = fetch_mldata('MNIST original', data_home='saved_mnist')



# set parameters (learning rate, regularization strength*, randomized initial weights)
learning_rate = .01 # .01 is commonly used
weight = np.zeros(10) # use randomly assigned ones (must be variable to avoid getting stuck in local min), what range to use???
bias = np.zeros(10) # initialize biases to zero?
inputs = np.zeros(10) # use import to incorporate inputs

# compute activations from inputs and weights
activation =  np.dot(np.transpose(weight),inputs) + bias

# pass activations through transfer function
apply_transfer = np.tanh(activation)

# calculate error/loss
error = None
gradient = None

# adjust parameters based on loss using backpropagation and GD