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
input_list = [] #list more useful when going through multiple image arrays
input_list.append(current_input)

possible_outputs = list(range(0, 10))

def activation(inputs):
    # activation_1 = np.zeros(4)
    print
    print 'length of input', len(inputs)
    print 'input'
    print np.shape(inputs)
    print inputs
    for i in range(0, len(inputs)):
        weight_1 = np.random.randint(0, high = 255, size = [10, 28])
        print 'weight'
        print np.shape(weight_1)
        print weight_1
        activation_1 =  np.dot(weight_1,inputs[i])
        print 'activation 1'
        print np.shape(activation_1)
        print activation_1
        #activation_1 is a 10x28 array

        apply_transfer_function_1 = np.tanh(activation_1)
        print 'apply transfer fx'
        print np.shape(apply_transfer_function_1)
        print apply_transfer_function_1
        #apply_transfer_function_1 is a 10x28 array with only 0s and 1s

        weight_2 = np.random.randint(0, high=255, size=[10, 10])
        print 'second weight'
        print np.shape(weight_2)
        print weight_2
        activation_2 = np.dot(np.transpose(weight_2),apply_transfer_function_1)
        activation_2[activation_2 < 400] = 0.0
        activation_2[activation_2 < 550] = 5.0
        activation_2[activation_2 < 650] = 6.0
        activation_2[activation_2 < 750] = 7.0
        activation_2[activation_2 < 850] = 8.0
        activation_2[activation_2 < 1000] = 9.0
        activation_2[activation_2 > 1000] = 10.0
        # need to come back and FIX THE FILTERING!!
        # activation_2[1000>activation_2 >= 800] = 8.0
        print 'activation 2'
        print np.shape(activation_2)
        print activation_2
        #activation_2 is a 28x28 array (back to original image dimensions)
        return activation_2
        # linear_classifier = np.sign(activation_2) #thresholding?
output_vector = [-1] * 10
print
print 'output vector', output_vector
output = random.choice(possible_outputs)
output_vector[output] = 1
print 'new output vector', output_vector
print
print 'predicted output', output
expected_output = 0

truth_vector = []
for i in possible_outputs:
    if i == 0:
        truth_vector.append(1)
    else:
        truth_vector.append(0)
print truth_vector

def error_computation(output_vector, truth_vector, activation_2):
   #  when does learning coefficient come into play / how to choose one, learning rate .001 (typical value used)

   error_vector = np.subtract(truth_vector, output_vector)
   error_vector = error_vector * .001
   error_vector_2 = [0] * 10
   print error_vector_2
   error_vector += error_vector_2
   print error_vector
   error_array = np.reshape(error_vector, (1,10))

   print 'errror array'
   print np.shape(error_array)
   print error_array
   print
   print 'error_vector', error_vector
   print
   print type(activation_2)
   print type(error_vector)
   # x

   correction_of_error = np.linalg.lstsq(output_vector, error_array)
   # this function is not working well
   return correction_of_error
   # print correction_of_error

def backpropagation():
    pass
# def gradient_descent_and_error_minimization(error):
#     gradient = np.gradient([error]) # dont need since lstq is doing the subtraction part
#     print gradient
# keep track of error while working backwards
# graph results to keep track?

a = activation(input_list)
error_correction_vector = error_computation(output_vector, truth_vector, a)


