import numpy as np
from sklearn.datasets import fetch_mldata
import random
saved_mnist = '/home/rubab/PycharmProjects/neural_network'
mnist = fetch_mldata('MNIST original', data_home='saved_mnist')
import sys

print np.shape(mnist.data)
print np.unique(mnist.target)
print
#
# # set parameters (learning rate, regularization strength*, randomized initial weights)
# learning_rate = .01 # .01 is commonly used - NOT IMPORTANT RIGHT NOW
# weight = np.zeros(10) # use randomly assigned ones (must be variable to avoid getting stuck in local min), what range to use???
# bias = np.zeros(10) # initialize biases to zero?
# inputs = np.zeros(10) # use import to incorporate inputs
#
# # compute activations from inputs and weights
# activation =  np.dot(np.transpose(weight),inputs) + bias
#
# # pass activations through transfer function
# apply_transfer = np.tanh(activation)
#
# # calculate error/loss
# error = None
# gradient = None
#
# # adjust parameters based on loss using backpropagation and GD

#Forward Propagation - using one hidden layer

subsample = mnist.data[0:5]
print np.shape(mnist.data)

current_input = mnist.data[0]
print np.shape(current_input)
print current_input

print
print np.shape(current_input)
print current_input

weight = np.random.randint(0, high = 255, size = [10, 784])
print 'weight'
print weight
print np.shape(weight)
print
print

# where does number of nodes come into play?
activation_1 =  np.dot(weight,current_input)
# print activation_1

apply_transfer_function_1 = np.tanh(activation_1)
# use as input of next layer
# print apply_transfer_function_1

activation_2 =  np.dot(np.transpose(weight),apply_transfer_function_1)
# no transfer function applied to outermost layer, start BP
# print activation_2