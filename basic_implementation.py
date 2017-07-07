import numpy as np
from sklearn.datasets import fetch_mldata
import random
from sklearn.metrics import mean_squared_error

saved_mnist = '/home/rubab/PycharmProjects/neural_network'
mnist = fetch_mldata('MNIST original', data_home='saved_mnist')



#Forward Propagation - using one hidden layer

subsample = mnist.data[0:5] #use subsample for running and training entire NN
current_input = mnist.data[0] #use current_input for building NN

# originally 784 row array, reshape tp 28x28 array
current_input = np.reshape(current_input, (28, 28))
print current_input #looks like a zero
input_list = []
input_list.append(current_input)

possible_outputs = list(range(0, 10))



def activation(inputs):
    activation_1 = 0
    print 'length of input', len(inputs)
    for input in range(0, len(inputs)):
        weight_1 = np.random.randint(0, high = 255, size = [10, 28])
        activation_1 +=  np.dot(weight_1,inputs[input])
        #activation_1 is a 10x28 array

        apply_transfer_function_1 = np.tanh(activation_1)
        #apply_transfer_function_1 is a 10x28 array with only 0s and 1s

        weight_2 = np.random.randint(0, high=255, size=[10, 28])
        activation_2 =  np.dot(np.transpose(weight_2),apply_transfer_function_1)
        #activation_2 is a 28x28 array (back to original image dimensions)

        linear_classifier = np.sign(activation_2) #thresholding?

output = [random.choice(possible_outputs)]
expected_output = [0]

def error_computation(output, expected_output):
   error = mean_squared_error(output, expected_output)
   print error
   return error

def gradient_descent():
    


activation(input_list)
error_computation(output, expected_output)