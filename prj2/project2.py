import numpy as np
import sys
import math
import matplotlib.pyplot as plt

"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""


# instantiate generator object with seed to pass around
seed = 12345
from numpy.random import default_rng
rng = default_rng(seed)









# A class which represents a single neuron
class Neuron:
    #initialize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr, weights=None):
        self.activation = activation
        self.input_num = input_num
        self.lr = lr

        # the last value of the weights vector is the bias
        self.weights = weights if weights is not None else np.random.rand(input_num+1)

        #initialize inputs and outputs arrays for saving feedforward values for later use in backprop
        self.inputs = None
        self.output = None

        # initialize partial derivatives for each weight
        self.partialderivatives = None
        
    #This method returns the activation of the net
    def activate(self,net):
        if self.activation == "logistic":
            post_activation = 1 / (1 + math.exp(-net))

        elif self.activation == "linear":
            post_activation = net

        else:
            print("Unrecognized activation function.")

        return post_activation
        
    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self, inputs):

        # Pre activation function matrix math (vectors -> scalar)
        net = np.sum(np.multiply(self.weights[:-1], inputs)) + self.weights[-1]

        # Output from neuron after activation function (scalar -> scalar)
        neuron_output = self.activate(net)

        # Save feedforward values for use in backprop later
        self.inputs = inputs
        self.output = neuron_output

        return neuron_output

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        if self.activation == "logistic":
            act_deriv = self.output * (1 - self.output)

        elif self.activation == "linear":
            act_deriv = 1

        else:
            print("Unrecognized activation function.")

        return act_deriv
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        
        # Assuming wtimesdelta is a lx1 vector of w x delta values from the next layer of size lx1
        actderiv = self.activationderivative()

        # (eqn 2 on summary page) (scalar)
        prev_delta = np.sum(wtimesdelta)*actderiv


        # (partial E / partial w)'s (for the weights, but not the bias)
        partial_derivatives = prev_delta*self.inputs

        # adding just the prev_delta term to the end since bias isn't multiplied by an input
        self.partialderivatives = np.append(partial_derivatives, prev_delta)

        # w * delta vector to be used in the previous layer (wtimesdelta passed back from this neuron)
        # last value is the bias which isn't really needed/used since there is no w*delta term needed for the bias
        prev_wtimesdelta = prev_delta*self.weights

        #prev_wtimesdelta[-1] = prev_delta   # Unnecessary code I originally included, shouldn't affect any calculations

        return prev_wtimesdelta
    
    #Simply update the weights using the partial derivatives and the learning weight
    def updateweight(self):

        # Weight update calculation (weights and bias) (eqn 4 on summary page)
        self.weights = self.weights - self.lr * self.partialderivatives
        
        return self.weights

        
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initialize randomly)
    def __init__(self, numOfNeurons, activation, input_num, lr, weights=None):
        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights if weights is not None else np.random.rand(numOfNeurons, input_num+1)  # 2D #cols is weights (weights + bias) & #rows are the  #neurons 

        # Create array of neurons in the layer
        self.neurons = []
        
        for i in range(self.numOfNeurons):

            _neuron_i = Neuron(self.activation, self.input_num, self.lr, self.weights[i,:])
            self.neurons.append(_neuron_i)
        
        
    #calculate the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calculate() method)      
    def calculate(self, input):
        
        neurons_outputs = np.zeros(self.numOfNeurons)

        # Calculate neurons' outputs then store their outputs
        for i in range(self.numOfNeurons):

            _output_i = self.neurons[i].calculate(input)

            neurons_outputs[i] = _output_i
        
        return neurons_outputs
        
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta, last=None):
        #wtimesdelta is a 2D matrix from the previous layer

        # 2D Matrix of wtimesdelta weights (each row is for a neuron's delta*w vector from this layer)
        prev_wtimes_delta = np.zeros([self.numOfNeurons, self.input_num+1])

        # Iterates through neurons top to bottom and calculates the wtimesdelta vector for each one
        for i in range(self.numOfNeurons):

            if last == "last":
                # loss derivatives are the first wtimesdelta values passed, so wtimesdelta is really a misnomer
                prev_wtimes_delta[i, :] = self.neurons[i].calcpartialderivative(wtimesdelta[i])
            else:
                # Update all other layers with w times delta values
                prev_wtimes_delta[i, :] = self.neurons[i].calcpartialderivative(wtimesdelta[:, i])
            #update the ith neuron
            self.neurons[i].updateweight()
        
        # If the w*deltas should be summed, it could happen here, summing column-wise
        return prev_wtimes_delta

# A class for a Convolutional Layer
class ConvolutionalLayer:
    # restricted to 2D convolutions , square kernels, stride = 1, padding = 'valid'
    def __init__(self, numKernels, kernelSize, activation, inputDim, lr, weights=None):
        # change inputDim name to inputSize? Is it the same as the other?
        self.numKernels = numKernels # same as number of filters
        self.kernelSize = kernelSize # will be an int since we assume it to be square
        self.activation = activation 
        self.inputDim = inputDim # np.array([m, n, p]) (doesn't have to be square), m = num input channels, n = height of input, p = width of input, 
        self.lr = lr
        self.weights = weights if weights is not None else np.random.rand(kernelSize, kernelSize, inputDim[0], numKernels)
        #print(f"self.weights print out: {self.weights}")
        #print(f"self.weights SHAPE print out: {self.weights.shape}")
        # unlike FCL, biases is its own attribute for a CL
        self.biases = np.random.rand(numKernels) # one for each kernel
    
        # needed for reshaping neuron weights and is the same as the neuron input dim (which may or may not be needed to be included)
        numweights_per_neuron = kernelSize * kernelSize * inputDim[0]
        
        #Create Neuron objects for entire Layer Shape will be:
        
        # Shape of the output & of the neurons. m x n x p: m = number of output channels (same as num of kernels); 
        #n = output #rows is ((h_i + (padding term here) - h_f) / s) + 1 (where i stands for input and f for filter); 
        #p = output #columns is ((w_i + (padding term here) - w_f) / s) + 1
        self.output_shape = np.array([numKernels, inputDim[1] - kernelSize + 1, inputDim[2] - kernelSize + 1])
        
        # Initialize neuron tensor about to be filled
        self.neurons = np.zeros((self.output_shape[0], self.output_shape[1], self.output_shape[2]), dtype=object)
        
        #For use in the calculate method/calculatewdeltas method
        self.output = np.zeros((self.output_shape[0], self.output_shape[1], self.output_shape[2]))
        
        # Create Neurons for an output channel at a time
        for i in range(self.output_shape[0]):
            # Create Neurons for a row of an output channel (feature map) (would need to change dimension here if padding = 'same' or stride was not equal to 1)
            for j in range(self.output_shape[1]):
                # Create Neurons for column of an output channel (feature map) (would need to change dimension here if padding = 'same' or stride was not equal to 1)
                for k in range(self.output_shape[2]):
                    
                    # neurons require 1D vector of weights, so they are reshaped from 4D (really 3D) to 1D
                    _neuron_weights = np.reshape(self.weights[:, :, :, i], numweights_per_neuron)
                    
                    # add bias to the end of the neuron weights
                    _neuron_weights = np.append(_neuron_weights, self.biases[i])
                    #print(_neuron_weights)
                    
                    # create a neuron with specified weights
                    self.neurons[i, j, k] = Neuron(self.activation, numweights_per_neuron, self.lr, _neuron_weights)
                    
                    
        print(self.neurons.shape)
    def calculate(self, input_):
        # All inputs will be assumed to be 2D numpy arrays
        
        ### Calculate Each Neuron's output and store it in the output attribute
        
        # Each Output Channel
        for i in range(self.output_shape[0]):
            # Each Row of Output
            for j in range(self.output_shape[1]):
                # Each Column of Output
                for k in range(self.output_shape[2]):
                    
                    # Get the correct portion of the input to apply kernel to
                    #print(f"input shape: {input_.shape}")
                    #print()
                    #print()
                    #print(f"count_row {count_row}")
                    #print(f"count_Row + kernelsize = {count_row + self.kernelSize}")
                    #print(f"count_col {count_col}")
                    #print(f"count_col + kernelsize = {count_col + self.kernelSize}")                  
                    #print()
                    #print()
                    
                    input_i = input_[:, j:(j + self.kernelSize), k:(k + self.kernelSize)]
                    #print(f"ith input is: {input_i}")
                    
                    # reshape 3D to 1D so it can be fed to Neuron class
                    input_i = np.reshape(input_i, input_i.shape[0]*input_i.shape[1]*input_i.shape[2])
                    
                    self.output[i, j, k] = self.neurons[i, j, k].calculate(input_i)
                    
        
        return self.output   # I don't think this will be needed
        
    def calculatewdeltas(self, sum_wtimesdelta):
        #wtimesdelta is a 2D matrix from the previous layer

        # 2D Matrix of wtimesdelta weights (each row is for a neuron's delta*w vector from this layer)
        prev_wtimes_delta = np.zeros([self.numOfNeurons, self.input_num+1])

        # Iterates through neurons top to bottom and calculates the wtimesdelta vector for each one
        for i in range(self.numOfNeurons):

            if last == "last":
                # loss derivatives are the first wtimesdelta values passed, so wtimesdelta is really a misnomer
                prev_wtimes_delta[i, :] = self.neurons[i].calcpartialderivative(wtimesdelta[i])
            else:
                # Update all other layers with w times delta values
                prev_wtimes_delta[i, :] = self.neurons[i].calcpartialderivative(wtimesdelta[:, i])
            #update the ith neuron
            self.neurons[i].updateweight()
        
        # If the w*deltas should be summed, it could happen here, summing column-wise
        return prev_wtimes_delta

class FlattenLayer:
    # restricted to 2D convolutions , square kernels, stride = 1, padding = 'valid'
    def __init__(self, input):
        #only needs to intialize with the input
        self.input = input
        self.FL_out = None
        self.unflat = None

    def calculate(self, input):
        self.FL_out = input.flatten()
        self.FL_out = np.reshape(self.FL_out, (len(self.FL_out), 1))
        return self.FL_out

    def calculatewdeltas(self, FL_out):
        # wtimesdelta is a 2D matrix from the previous layer
        re_flat = np.reshape(self.FL_out, (len(self.FL_out),))
        self.unflat = re_flat.reshape(self.input.shape)

        return self.unflat

# class MaxPoolingLayer:
#     def __init__(self, kernal, inputDim):
#         self.kernal = kernal
#         self.inputDim = inputDim
#         self.mask = None
#     def calculate(self, input):
#
#
#     def calculatewdeltas(self, FL_out):
#         # wtimesdelta is a 2D matrix from the previous layer

























#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    #def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
    def __init__(self, inputSize, loss, lr):
         # inputSize can now be multi-dimensional now that we have convolutional layers
         self.inputSize = inputSize
         self.loss = loss
         self.lr = lr
         self.layers = []
         
    # adds a layer to the neural network
    def addLayer(self, layerType, numOfNeurons, activation, weights=None):
        
        # Decide which Layer to create FullyConnected, Convolutional, MaxPooling, or Flatten
        if layerType == "FCL":
            _new_layer = FullyConnected(numOfNeurons, activation, self.inputSize, self.lr, weights)
            
        elif layerType == "CL":
            # _new_layer = ConvolutionalLayer(numOfNeurons, activation, self.inputSize, self.lr, weights)
            pass
        elif layerType == "MPL":
            pass
        elif layerType == "FL":
            pass
        else:
            print("Unrecognized layer type.")
        
        # Change input size to size of new layer's output (same as numOfNeurons)
        self.inputSize = numOfNeurons
        
        # Add layer to NeuralNetwork
        self.layers.append(_new_layer)
        
        
    #Given an input, calculate the output (using the layers calculate() method) and return the last layer's output
    def calculate(self,input):
        
        layers_output = []

        # Update each layer's weights
        for i in range(self.numOfLayers):

            if i == 0:
                # First layer uses the input given by the user
                _layer_output_i = self.layers[i].calculate(input)
            
            else:
                # Subsequent layers use the output of previous layers as inputs
                _layer_output_i = self.layers[i].calculate(layers_output[i-1])
            
            # Need this for creating the 2D matrix of all NN layer outputs
            layers_output.append(_layer_output_i)
            
        #return np.reshape(layers_output[-1], [len(layers_output[-1]), 1])
        return layers_output[-1]


    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):

        loss_calcs = np.zeros((y.shape[0], y.shape[1]))
        if self.loss == "square error":

            for i in range(len(y)):
                loss_calcs[i] = 0.5 * np.square(np.subtract(y[i], yp[i]))


        elif self.loss == "binary cross entropy":
            for i in range(len(y)):
                loss_calcs[i] = (1 / len(yp)) * -((y[i] * math.log(yp[i])) + (1 - y[i]) * math.log(1 - yp[i]))

        else:
            print("Unrecognized loss function.")
    
        loss_calc_sum = np.sum(loss_calcs)

        return loss_calc_sum
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self, yp, y):
        
        # changed to 1D from submission code, and changed output form (i.e. shape of y)
        loss_derivs = np.zeros(len(y))
        
        if self.loss == "square error":
            
            for i in range(len(y)):

                loss_derivs[i] = (yp[i] - y[i])

        elif self.loss == "binary cross entropy":

            for i in range(len(y)):
                loss_derivs.append( -(y[i] / yp[i]) + ((1 - y[i]) / (1 - yp[i])))

        else:
            print("Unrecognized loss function.")
        
        # Changed to transpose from submission code
        return loss_derivs.T
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):

        yp = self.calculate(x)

        loss_deriv = self.lossderiv(yp, y)
        #print(loss_deriv.shape)

        next_wdeltas = []
        
        # Iterate through layers backwards calculating w*delta arrays for previous layers and updating the weights
        for i, e in reversed(list(enumerate(self.layers))):
            # i goes len(layers) - 1, len(layers) - 2,..., 1, 0
            if i == len(self.layers) - 1:
                # Get wdeltas differently for the last layer b/c wtimesdeltas given is a vector of the loss derivatives
                next_wdeltas.append(self.layers[i].calcwdeltas(loss_deriv, "last"))
                
            else:
                # iterate through the rest of the layers normally
                next_wdeltas.append(self.layers[i].calcwdeltas(next_wdeltas[len(self.layers) - (i + 2)]))

        return np.reshape(yp, (len(yp), 1))



# To run the code specify three arguments: learning_rate demonstration_name number_of_epochs
# for example: 0.5 example 1000
#          or: 1 and 1000
#          or: 0.1 xor 1000


if __name__ == "__main__":
    if (len(sys.argv) < 3):
        # input = np.array([[[1, 2, 3],
        #                     [4, 5, 6],
        #                     [7, 8, 9]],
        #                     [[-1, -2, -3],
        #                     [-5, -6, -7],
        #                     [-9, -10, -11]]])

        ###### Test code for flatten layer ###########
        input = np.array([[[1, 2, 3, 4],
                                   [5, 6, 7, 8],
                                   [9, 10, 11, 12],
                                   [13, 14, 15, 16]],
                                   [[-1, -2, -3, -4],
                                   [-5, -6, -7, -8],
                                   [-9, -10, -11, -12],
                                   [-13, -14, -15, -16]]])
        print(input)
        FL = FlattenLayer(input)
        output = FL.calculate(input)
        print(output)

        unflattten = FL.calculatewdeltas(output)
        print(unflattten )
        ###### Test code for flatten layer ###########


    elif (sys.argv[2] == 'example1'):
        from tensorflowtest_example1 import example1
        from parameters import generateExample1

        l1k1, l1b1, l2, l2b, input, output = generateExample1()
        example1()

    elif (sys.argv[2] == 'example2'):
        from tensorflowtest_example2 import example2
        from parameters import generateExample2

        l1k1, l1k2, l1b1, l1b2, l2c1, l2c2, l2b, l3, l3b, input, output = generateExample2()
        example2()

    elif (sys.argv[2] == 'example3'):
        from tensorflowtest_example3 import example3
        from parameters import generateExample3

        l1k1, l1k2, l1b1, l1b2, l3, l3b, input, output = generateExample3()
        example3()


# if __name__=="__main__":
#     # Testing Code
#         if (len(sys.argv)<3):
#
#             # args: inputSize, loss, lr (check if NeuralNetwork class works after changing config --check)
#             example_network = NeuralNetwork(10, "square error", float(sys.argv[1]))
#
#             # layerType, numOfNeurons, activation, weights=None (check if addLayer is working for FCL case --check)
#             example_network.addLayer("FCL", 5, "logistic")
#
#
#             # Check if addLayer method is initializing weights correctly when none are given for FCL case --check
#             for i, lays in enumerate(example_network.layers):
#                 for j, neurs in enumerate(lays.neurons):
#                     #print(f"Layer {i + 1} Neuron {j + 1} ")
#                     #print(lays.neurons[j].weights[:])
#                     pass
#
#             # Check if update to inputSize is working within the NN class for FCL case --check
#             example_network.addLayer("FCL", 5, "logistic")
#
#             for i, lays in enumerate(example_network.layers):
#                 for j, neurs in enumerate(lays.neurons):
#                     #print(f"Layer {i + 1} Neuron {j + 1} ")
#                     #print(lays.neurons[j].weights[:])
#                     pass
#
#             # Check if ConvolutionalLayer class is creating weights correctly
#             # numKernels, kernelSize, activation, inputDim, lr, weights=None
#             #conv_lay = ConvolutionalLayer(2, 3, "logistic", np.array([5, 5, 3]), 0.01)   # example in notes
#             conv_lay = ConvolutionalLayer(1, 2, "logistic", np.array([2, 4, 4]), 0.01)
#             # Check if input_ is reshaped correctly in CL calculate method
#             CL_test_input = np.array([[[1, 2, 3, 4],
#                                        [5, 6, 7, 8],
#                                        [9, 10, 11, 12],
#                                        [13, 14, 15, 16]],
#                                       [[-1, -2, -3, -4],
#                                        [-5, -6, -7, -8],
#                                        [-9, -10, -11, -12],
#                                        [-13, -14, -15, -16]]])
#             CL_test_input = np.array([[[1, 2, 3, 4],
#                                        [5, 6, 7, 8],
#                                        [9, 10, 11, 12],
#                                        [13, 14, 15, 16]],
#                                       [[-1, -2, -3, -4],
#                                        [-5, -6, -7, -8],
#                                        [-9, -10, -11, -12],
#                                        [-13, -14, -15, -16]]])
#             print(f" test input size: {CL_test_input.shape}")
#
#             output = conv_lay.calculate(CL_test_input)
#
#
#             print(output.shape)
#
#
#
#
#
#
#     elif (sys.argv[2]=='example'):
#
#         num_epochs = int(sys.argv[3])
#
#         print(f"Run example from class (single step) for {num_epochs} epochs.")
#
#         w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
#         x=np.array([[0.05, 0.1]])
#         y=np.array([[0.01, 0.99]])
#
#         # params: self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None
#         #example_network = NeuralNetwork(2, np.array([2, 2]), 2, ["logistic", "logistic"], "square error", float(sys.argv[1]), w)
#
#         # example code to show the netwrok can scale
#         example_network = NeuralNetwork(3, np.array([10,10, 2]), 2, ["logistic", "logistic", "logistic"], "square error", float(sys.argv[1]))
#
#         #extra
#         losses = []
#
#
#         # Train the network, each epoch goes through all of the datapoints
#         # For example: if num_epochs = 100 & there are 10 training datapoints (i.e len(y) = 10) then the weights are updated 10 * 100 = 1,000 times
#         for i in range(num_epochs):
#
#             # extra
#             yp = np.zeros((2,1))
#
#
#             for i in range(len(y)):
#                 # Train the network for 1 datapoint
#                 network_output = example_network.train(x[i], y[i])
#                 #print(network_output)
#                 # store the prediction for the datapoint
#                 yp[:, i] = network_output.T
#
#             # Append the loss from the epoch
#             losses.append(example_network.calculateloss(yp.T, y))
#
#
#         print(f"Network final predictions {yp[:, -1]}")
#         print("Expected Output: [0.01, 0.99]")
#
#         ### Plot the convergence
#
#         plt.figure(dpi=150)
#         plt.plot(range(len(losses)), losses)
#
#         #figure formatting
#         plt.title("Convergence Curve: Example")
#         plt.xlabel("Epochs")
#         plt.ylabel("Error")
#
#         plt.show()
#
#
#
#         # Printing the weights going Layers left to right & Neurons top to bottom
#         for i, lays in enumerate(example_network.layers):
#
#             for j, neurs in enumerate(lays.neurons):
#                 print(f"Layer {i + 1} Neuron {j + 1} ")
#                 print(lays.neurons[j].weights[:])
#
#
#
#
#     elif(sys.argv[2]=='and'):
#
#         num_epochs = int(sys.argv[3])
#
#         print(f"Training AND gate for {num_epochs} epochs.")
#
#
#         and_inputs = np.array([[0, 0],
#                                [0, 1],
#                                [1, 0],
#                                [1, 1]])
#
#         and_outputs = np.array([[0],
#                                 [0],
#                                 [0],
#                                 [1]])
#
#
#         ### Initiate the Neural Network
#
#         #test_weights = np.array([np.array([[1, 1, 5]])])
#
#         # params: numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None
#         SLP_network = NeuralNetwork(1, np.array([1]), 2, ["logistic"], "square error", float(sys.argv[1]))
#
#
#         ### Train the Neural Network
#
#         losses = []
#
#         for i in range(num_epochs):
#
#             # vector of predictions for each datapoint in the dataset
#             yp = np.zeros((len(and_outputs),1))
#
#             # train network using whole dataset (1 epoch) and update weights each iteration for the datapoint being used
#             for i in range(len(and_outputs)):
#
#                 network_output = SLP_network.train(and_inputs[i], and_outputs[i])
#
#                 # store the prediction for the datapoint
#                 yp[i] = network_output
#
#             # Append the loss from the epoch
#             losses.append(SLP_network.calculateloss(yp, and_outputs))
#
#
#         print(f"Converging network final predictions {yp[:, -1]}")
#         print("Expected Output: [0, 0, 0, 1]")
#
#         ### Plot the convergence
#
#         plt.figure(dpi=150)
#         plt.plot(range(len(losses)), losses)
#
#         #figure formatting
#         plt.title("Convergence Curves: AND Gate")
#         plt.xlabel("Epochs")
#         plt.ylabel("Error")
#
#         plt.show()
#
#     elif(sys.argv[2]=='xor'):
#
#         num_epochs = int(sys.argv[3])
#
#         print(f"Training XOR gate for {num_epochs} epochs.")
#
#         xor_inputs = np.array([[0, 0],
#                                [0, 1],
#                                [1, 0],
#                                [1, 1]])
#         xor_outputs = np.array([[0],
#                                 [1],
#                                 [1],
#                                 [0]])
#
#
#         ### Initialize the Network
#
#         # 5 x 1 network -> converges
#         converging_network = NeuralNetwork(2, np.array([5, 1]), 2, ["logistic", "logistic"], "square error", float(sys.argv[1]))
#
#
#         # Single Perceptron -> does NOT converge
#         SLP_network = NeuralNetwork(1, np.array([1]), 2, ["logistic"], "square error", float(sys.argv[1]))
#
#
#
#         ### Train the Neural Network
#
#         # Lists for storing the losses at end of each epoch for plotting
#         losses_conv= []
#         losses_nconv = []
#
#         for j in range(num_epochs):
#
#             # vector of predictions for each datapoint in the dataset
#             yp_conv = np.zeros((len(xor_outputs),1))
#
#             yp_nconv = np.zeros((len(xor_outputs),1))
#             # train network using whole dataset (1 epoch) and update weights each iteration for the datapoint being used
#             for i in range(len(xor_outputs)):
#
#                 # Train both networks
#                 print(f"input shape {xor_inputs[i].shape}")
#                 print(f"input shape {xor_outputs[i].shape}")
#                 converging_network_output = converging_network.train(xor_inputs[i], xor_outputs[i])
#                 SLP_network_output = SLP_network.train(xor_inputs[i], xor_outputs[i])
#
#                 # store the prediction for the datapoint for each network
#                 yp_conv[i] = converging_network_output
#                 yp_nconv[i] = SLP_network_output
#
#
#
#             # Append the loss from the epoch to know when to stop the while loop
#             losses_conv.append(converging_network.calculateloss(yp_conv, xor_outputs))
#             losses_nconv.append(SLP_network.calculateloss(yp_nconv, xor_outputs))
#
#
#         print(f"Converging network final predictions {yp_conv[:, -1]}")
#         print(f"Not Converging SLP final predictions {yp_nconv[:, -1]}")
#         print("Expected Output: [0, 1, 1, 0]")
#
#
#         ### Plot the convergence
#
#
#         plt.figure(dpi=150)
#         plt.plot(range(len(losses_conv)), losses_conv)
#         plt.plot(range(len(losses_nconv)), losses_nconv)
#
#
#         #figure formatting
#         plt.title("Convergence Curves: XOR Gate")
#         plt.xlabel("Epochs")
#         plt.ylabel("Error")
#         plt.legend(["Converging Network [5 x 1]", "Not Converging SLP"])
#
#         plt.show()
