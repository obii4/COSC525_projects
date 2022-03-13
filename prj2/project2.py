import numpy as np
import sys
import math


# function used for padding from numpy website: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 111)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    

# A class which represents a single neuron
class Neuron:
    #initialize neuron with activation type, number of inputs, learning rate, and possibly with set params
    def __init__(self, activation, input_num, lr, params=None):
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.params = params if params is not None else np.random.rand(input_num+1) # the last value of the params vector is the bias
        self.inputs = None   #initialize inputs array for saving feedforward values for later use in backprop
        self.output = None   #initialize outputs arrays for saving feedforward values for later use in backprop
        self.partialderivatives = None  # initialize partial derivatives for each weight
        
    #This method returns the activation of the net
    def activate(self, net):
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
        net = np.sum(self.params[:-1]*inputs) + self.params[-1]
        
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
        prev_delta = wtimesdelta*actderiv

        # (partial E / partial w)'s (for the params, but not the bias)
        partial_derivatives = prev_delta*self.inputs

        # adding just the prev_delta term to the end since bias isn't multiplied by an input
        self.partialderivatives = np.append(partial_derivatives, prev_delta)

        # w * delta vector to be used in the previous layer (wtimesdelta passed back from this neuron)
        # last value is the bias which isn't really needed/used since there is no w*delta term needed for the bias
        prev_wtimesdelta = prev_delta*self.params
        
        return prev_wtimesdelta
        
    def updateweightCL(self):
        
        # Need to multiply by activation function derivative here since 
        self.partialderivatives = self.partialderivatives*self.activationderivative()
        
        # parameters update calculation (weights and bias)
        self.params = self.params - self.lr * self.partialderivatives
        
        return self.params

    #Simply update the weights using the partial derivatives and the learning weight
    def updateweight(self):
        
        # Weight update calculation (weights and bias) (eqn 4 on summary page)
        self.params = self.params - self.lr * self.partialderivatives

        return self.params
        
        
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initialize randomly)
    def __init__(self, numOfNeurons, activation, input_num, lr, weights=None):
        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights if weights is not None else np.random.rand(numOfNeurons, input_num+1)  # 2D #cols is weights (weights + bias) & #rows are the #neurons 
        self.id = "FCL"
        
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
    def calcwdeltas(self, wtimesdelta):
        #wtimesdelta is a 2D matrix from the previous layer

        # 2D Matrix of wtimesdelta weights (each row is for a neuron's delta*w vector from this layer)
        prev_wtimes_delta = np.zeros([self.numOfNeurons, self.input_num+1])
        
        # Iterates through neurons top to bottom and calculates the wtimesdelta vector for each one
        for i in range(self.numOfNeurons):

            # Update a layer with w times delta values (sum of w times delta -scalar- since it's being fed to a neuron) from previous layer
            prev_wtimes_delta[i, :] = self.neurons[i].calcpartialderivative(wtimesdelta[i])

            #update the ith neuron
            self.neurons[i].updateweight()

        # columnwise sum to get a 1D array of wtimesdelta sums for each neuron of the next layer to be updated
        prev_wtimes_delta = np.sum(prev_wtimes_delta, axis=0)

        return prev_wtimes_delta

# A class for a Convolutional Layer
class ConvolutionalLayer:
    # restricted to 2D convolutions , square kernels, stride = 1, padding = 'valid'
    def __init__(self, numKernels, kernelSize, activation, inputDim, lr, weights=None, biases=None):
        # change inputDim name to inputSize? Is it the same as the other?
        self.numKernels = numKernels # same as number of filters
        self.kernelSize = kernelSize # will be an int since we assume it to be square
        self.activation = activation 
        self.inputDim = inputDim # np.array([m, n, p]) (doesn't have to be square), m = num input channels, n = height of input, p = width of input, 
        self.lr = lr
        self.weights = weights if weights is not None else np.random.rand(numKernels, inputDim[0], kernelSize, kernelSize)
        self.biases = biases if biases is not None else np.random.rand(numKernels) # one for each kernel
        self.input = None
        self.id = "CL"
        
        # Shape of the output & of the neurons. m x n x p: m = number of output channels (same as num of kernels); 
        #n = output #rows is ((h_i + (padding term here) - h_f) / s) + 1 (where i stands for input and f for filter/kernel); 
        #p = output #columns is ((w_i + (padding term here) - w_f) / s) + 1
        self.output_shape = np.array([numKernels, inputDim[1] - kernelSize + 1, inputDim[2] - kernelSize + 1])
        
        self.neurons = np.zeros((self.output_shape[0], self.output_shape[1], self.output_shape[2]), dtype=object)   # Initialize neuron tensor about to be filled
        self.output = np.zeros((self.output_shape[0], self.output_shape[1], self.output_shape[2]))  #For use in the calculate method/calculatewdeltas method
        self.sum_actderiv = 0.0   #sum of derivative of activation function needed for backprop
        
        # needed for reshaping neuron weights
        numweights_per_neuron = kernelSize * kernelSize * inputDim[0]
        
        # Create Neurons for an output channel at a time
        for i in range(self.output_shape[0]):
            # Create Neurons for a row of an output channel (feature map) (would need to change dimension here if padding = 'same' or stride was not equal to 1)
            for j in range(self.output_shape[1]):
                # Create Neurons for column of an output channel (feature map) (would need to change dimension here if padding = 'same' or stride was not equal to 1)
                for k in range(self.output_shape[2]):
                    
                    # neurons require 1D vector of weights, so they are reshaped from 4D (really 3D) to 1D
                    _neuron_weights = np.reshape(self.weights[i, :, :, :], numweights_per_neuron)
                    
                    # add bias to the end of the neuron weights
                    _neuron_weights = np.append(_neuron_weights, self.biases[i])
                    
                    # create a neuron with specified weights
                    self.neurons[i, j, k] = Neuron(self.activation, numweights_per_neuron, self.lr, _neuron_weights)
                    
                    
    def calculate(self, input_):
        
        self.input= input_   # All inputs will be assumed to be 3D numpy arrays
        
        ### Calculate Each Neuron's output and store it in the output attribute
        for i in range(self.output_shape[0]):   # Each Output Channel
            for j in range(self.output_shape[1]):   # Each Row of Output
                for k in range(self.output_shape[2]):   # Each Column of Output
                    
                    input_i = input_[:, j:(j + self.kernelSize), k:(k + self.kernelSize)]
                    
                    # reshape 3D to 1D so it can be fed to Neuron class
                    input_i = np.reshape(input_i, input_i.shape[0]*input_i.shape[1]*input_i.shape[2])
                    
                    self.output[i, j, k] = self.neurons[i, j, k].calculate(input_i)
                    
                    # rolling sum of derivative of activation functions for backprop
                    self.sum_actderiv = self.sum_actderiv + self.neurons[i, j, k].activationderivative()
        
        return self.output
        
    def calcwdeltas(self, wtimesdelta):
        # wtimesdelta is a 3D matrix from the previous layer m x n x p: m = num channels; n = 2D delta matrix height; p = 2D delta matrix width
        # one delta for each neuron
        
        # 4D Matrix of partials to update weights
        partial_derivatives = np.zeros([self.numKernels, self.inputDim[0], self.kernelSize, self.kernelSize])
        
        ######################## Find partial derivatives of each weight ##################################################
        
        # cross-correlation of input with delta's to calculate the (partial E / partial w)'s partial derivatives for each weight w
        for h in range(self.numKernels):   # iterate through wtimesdelta channels. I had wtimesdelta.shape[0] before which should be the same
            # Each Output Channel  
            for i in range(self.inputDim[0]):   # num of input channels (Don't need?)
                # Each Row of Output
                for j in range(self.kernelSize):   # kernelSize = (input_height - output_height / stride) + 1 so we use kernelSize for simplicity
                    # Each Column of Output
                    for k in range(self.kernelSize):   # kernelSize = (input_width - output_width / stride) + 1 so we use kernelSize for simplicity
                    
                        #find correct portion 3D portion of the input and flatten it  
                        input_ij = np.reshape(self.input[i, j:(j + self.output_shape[1]), k:(k + self.output_shape[2])], self.output_shape[1]*self.output_shape[2])
                        wtimesdelta_ij = np.reshape(wtimesdelta[h, :, :], self.output_shape[1]*self.output_shape[2])
                        
                        #calc partial derivative for a single weight and add it to the matrix
                        partial_derivatives[h, i, j, k] = np.dot(input_ij, wtimesdelta_ij)


        ###################### Calculate (partial E / partial out)'s or "wtimesdelta" for previous layer #################
        
        # initialize partial E wrt input (same size as input)
        wtimesdelta_prev = np.zeros((self.inputDim[0], self.inputDim[1], self.inputDim[2]))
        
        # add padding to wtimesdelta
        padding_for_a_side = self.inputDim[1] - self.output_shape[1]  # calculate padding needed for a side (Note inputs/outputs assumed to be square here so 1 is arbitrary - could be 2)
        padded_wtimesdelta = np.zeros((wtimesdelta.shape[0], wtimesdelta.shape[1] + 2*padding_for_a_side, wtimesdelta.shape[2] + 2*padding_for_a_side))
        
        # call pad_with function at top of script, use padding values of zero
        for i in range(wtimesdelta.shape[0]):
            padded_wtimesdelta[i, :, :] = np.pad(wtimesdelta[i, :, :], padding_for_a_side, pad_with, padder=0)
        
                            
        # iterate through each kernel
        for h in range(self.numKernels):
            # iterate through a kernel channel (number of kernel channels equal to number of input channels)
            for i in range(self.inputDim[0]):
                # iterate through padded wtimesdelta 2D convolved with kernel (apply transpose of kernel across the padded delta matrix)
                for j in range(self.inputDim[1]):
                    for k in range(self.inputDim[2]):
                        
                        # I arbitrarily chose the 0,0 neuron since they all neurons in this kernel have shared weights I could choose any of them, I choose 0,0 because at a minimum my output will be a 1x1 with indices 0-0
                        # get kernel h and reshape to 3D 
                        kernel = np.reshape(self.neurons[h, 0, 0].params[:-1], (self.inputDim[0], self.kernelSize, self.kernelSize))
                        
                        # get channel i of kernel h and transpose it
                        kernel_channel = np.transpose(kernel[i, :, :])
   
                        # add calculation to correct spot in wtimesdelta_prev (Note the calculations are summed across the kernels)
                        wtimesdelta_prev[i, j, k] = wtimesdelta_prev[i, j, k] + np.tensordot(kernel_channel, padded_wtimesdelta[h, j:(j + self.kernelSize), k:(k + self.kernelSize)], axes=2)*self.sum_actderiv
        
                
        ####################### Update the weights. Iterate through all neurons and use corresponding partial derivatives (flattened) to update the weights

        # Iterate through all neurons
        for i in range(self.output_shape[0]):           # Each Output Channel
            for j in range(self.output_shape[1]):       # Each Row of Output
                for k in range(self.output_shape[2]):   # Each Column of Output
                    
                    # pass back flattened partial derivatives
                    self.neurons[i, j, k].partialderivatives = np.reshape(partial_derivatives[i, :, :, :], self.inputDim[0]*self.kernelSize*self.kernelSize)
                    
                    # add partial derivative of bias (append the product of activation function derivatives from all neurons and the sum of all "wtimesdelta" to the end)
                    self.neurons[i, j, k].partialderivatives = np.append(self.neurons[i, j, k].partialderivatives, self.sum_actderiv*np.sum(wtimesdelta))
                    
                    #update weights with new partial derivatives
                    self.neurons[i, j, k].updateweightCL()
        

        return wtimesdelta_prev


class FlattenLayer:
    # restricted to 2D convolutions , square kernels, stride = 1, padding = 'valid'
    def __init__(self):
        self.input = None
        self.FL_out = None
        self.id = "FL"

    def calculate(self, input_):
        self.input = input_
        self.FL_out = input_.flatten()
        self.FL_out = np.reshape(self.FL_out, (1, len(self.FL_out)))

        return self.FL_out

    def calcwdeltas(self, wtimesdelta):

        # wtimesdelta is a 2D matrix from the previous layer
        re_flat = np.reshape(wtimesdelta[:-1], (len(wtimesdelta[:-1]), 1))

        unflat = re_flat.reshape(self.input.shape)

        return unflat




class MaxPoolingLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size #square only
        self.stride = pool_size #always equal to pool
        self.output = None #output of maxpooling layer
        self.input_ = None
        self.id = "MPL"

    def calculate(self, input_):
        
        self.input_ = input_
        
        self.output = np.zeros((input_.shape[0], input_.shape[1] // self.pool_size, input_.shape[2] // self.pool_size))

        for i in range(self.output.shape[0]):
            for j in range(self.output.shape[1]):
                for k in range(self.output.shape[2]):
                    self.output[i, j, k] = input_[i, j: j * self.pool_size + self.stride, k: k * self.pool_size + self.stride].max()

        return self.output #, self.mask

    def calcwdeltas(self, wtimesdelta):
        
        # initialize wtimesdelta from this layer to next
        prev_wtimesdelta = np.zeros(self.input_.shape)
        
        # iterate through every wtimesdelta value to determine where it should go in the previous layer's wtimesdelta
        for i in range(wtimesdelta.shape[0]):
            for j in range(wtimesdelta.shape[1]):
                for k in range(wtimesdelta.shape[2]):
                    
                    # iterate through every element of input channel corresponding to wtimesdelta channel to see if it equals to the max found earlier
                    for g in range(self.input_.shape[1]):
                        for h in range(self.input_.shape[2]):
                            
                            # Assign the delta value to new delta value location if it is in the correct spot
                            if self.input_[i, g, h] == self.output[i, j, k]:
                                prev_wtimesdelta[i, g, h] = wtimesdelta[i, j, k]              

        return prev_wtimesdelta



#An entire neural network        
class NeuralNetwork:
    def __init__(self, inputSize, loss, lr):
         self.inputSize = inputSize   # inputSize is now 3D to start now that we have convolutional layers
         self.loss = loss
         self.lr = lr
         self.layers = []
         self.numOfLayers = 0
         
    # adds a layer to the neural network
    def addLayer(self, layerType, activation=None, numOfNeurons=None, numKernels=None, kernelSize=None, pool_size=None, weights=None, biases=None):
        
        # Decide which Layer to create FullyConnected, Convolutional, MaxPooling, or Flatten
        if layerType == "FCL":
            
            _new_layer = FullyConnected(numOfNeurons, activation, self.inputSize, self.lr, weights)
            
            #  Change input size for next layer to output size of this layer (same as numOfNeurons for a FullyConnected layer)
            self.inputSize = numOfNeurons
            
        elif layerType == "CL":
             # self, numKernels, kernelSize, activation, inputDim, lr, weights=None
             _new_layer = ConvolutionalLayer(numKernels, kernelSize, activation, self.inputSize, self.lr, weights, biases)
             
             #  Change input size for next layer to output size of this layer (still a 3D shape after a ConvolutionalLayer)
             self.inputSize = (numKernels, self.inputSize[1] - kernelSize + 1, self.inputSize[2] - kernelSize + 1)
             
        elif layerType == "MPL":
            # self, pool_size
             _new_layer = MaxPoolingLayer(pool_size)
             
             #  Change input size for next layer to output size of this layer (same number of channels, but height and width divided by pool_size)
             self.inputSize = (self.inputSize[0], self.inputSize[1] // pool_size, self.inputSize[2] // pool_size)
        
        elif layerType == "FL":
            
            _new_layer = FlattenLayer()
            
            # Change input size for next layer to output size of this layer (ASSUME going from 3D convlayer output to 1D fullyconnlayer input)
            self.inputSize = self.inputSize[0]*self.inputSize[1]*self.inputSize[2]
        else:
            print("Unrecognized layer type.")
        
        # Add layer to NeuralNetwork
        self.layers.append(_new_layer)
        
        # keep count of number of layers for later when iterating through and updating layers
        self.numOfLayers +=1
        
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
                
                #####NOTE: In order to get correct numbers a scaling of 2 is needed here to agree with tensorflow code (only for CNN) ####
                #loss_derivs[i] = (yp[i] - y[i])   # original code
                loss_derivs[i] = 2*(yp[i] - y[i])

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
                wdeltas = self.layers[i].calcwdeltas(loss_deriv)
                next_wdeltas.append(wdeltas)
                #print(f"WTIMESDELTA 1: {wdeltas}")
            else:
                # iterate through the rest of the layers normally
                wdeltas = self.layers[i].calcwdeltas(next_wdeltas[len(self.layers) - (i + 2)])
                next_wdeltas.append(wdeltas)
                #print(f"WTIMESDELTA: {wdeltas}")

        return np.reshape(yp, (len(yp), 1))



# To run the code specify the example to be run
# for example: example1
#          or: example2
#          or: example3

if __name__=="__main__":
    # Testing Code
    
    if (sys.argv[1] == 'example1'):
        from tensorflowtest_example1 import example1
        from parameters import generateExample1
        
        num_epochs = 1


        ################# Get weights, biases, input, output ##################


        l1k1, l1b1, l2, l2b, input_, output = generateExample1()
        
        # reshape some generated values
        l1b1 = l1b1.reshape(1,1)
        input_ = input_.reshape((1, 5, 5))
        l2_params = np.concatenate((l2, l2b.reshape(1,1)), axis=1)
        
        
        ################# Construct Network and Run ###########################
        
        
        # self, inputSize, loss, lr
        example1_NN = NeuralNetwork((1, 5, 5), "square error", 100.0)
        
        # self, layerType, activation, numOfNeurons=None, numKernels=None, kernelSize=None, weights=None
        example1_NN.addLayer("CL", activation="logistic", numKernels=1, kernelSize=3, weights=np.array([[l1k1]]), biases=l1b1)
        example1_NN.addLayer("FL")
        example1_NN.addLayer("FCL", "logistic", numOfNeurons=1, weights=l2_params)
        
        # Train the network, each epoch goes through all of the datapoints
        for i in range(num_epochs):
            network_output = example1_NN.train(input_, output)
            
            
        ################# Print Results#######################################
        print("##################CUSTOM NETWORK RESULTS######################")
        
        print(f"Network output: {network_output}")
        
        # Printing the weights going Layers left to right & Neurons top to bottom
        for i, lays in enumerate(example1_NN.layers):

            # Print neuron weights depending on the type of layer they are in            
            if lays.id == "CL":
                
                for j in range(lays.numKernels):
                    # print weights of 1 neuron from a kernel channel
                    print()
                    print(f"Convolutional Layer {i}, Kernel {j} Weights")
                    print(np.reshape(lays.neurons[j, 0, 0].params[:-1], (lays.inputDim[0], lays.kernelSize, lays.kernelSize)))
                    print()
                    print(f"Convolutional Layer {i}, Kernel {j} Bias")
                    print(lays.neurons[j, 0, 0].params[-1])
                    print()
            elif lays.id == "FCL":
                
                for j in range(len(lays.neurons)):
                    print()
                    print(f"Fully Connected Layer {i} Neuron {j} Weights")
                    print(lays.neurons[j].params[:-1])
                    print()
                    print(f"Fully Connected Layer {i} Neuron {j} Bias")
                    print(lays.neurons[j].params[-1])
            else:
                pass
        
        # TensorFlow Results
        print("####################TENSORFLOW RESULTS#########################")
        example1()

    elif (sys.argv[1] == 'example2'):
        from tensorflowtest_example2 import example2
        from parameters import generateExample2
        
        num_epochs = 1
        
        
        ################# Get weights, biases, input, output ##################
        
        
        l1k1, l1k2, l1b1, l1b2, l2c1, l2c2, l2b, l3, l3b, input_, output = generateExample2()
        
        # reshape some generated values
        input_ = input_.reshape((1, 7, 7))
        l3_params = np.concatenate((l3, l3b.reshape(1,1)), axis=1)
        l2b = l2b.reshape(1,1)
        
        
        ################# Construct Network and Run ###########################
        
        
        # self, inputSize, loss, lr
        example2_NN = NeuralNetwork((1, 7, 7), "square error", 100.0)
        
        # self, layerType, activation, numOfNeurons=None, numKernels=None, kernelSize=None, weights=None
        example2_NN.addLayer("CL", activation="logistic", numKernels=2, kernelSize=3, weights=np.array([[l1k1], [l1k2]]), biases=np.array([l1b1, l1b2]))
        example2_NN.addLayer("CL", activation="logistic", numKernels=1, kernelSize=3, weights=np.array([[l2c1, l2c2]]), biases=l2b)
        example2_NN.addLayer("FL")
        example2_NN.addLayer("FCL", "logistic", numOfNeurons=1, weights=l3_params)
        
        # Train the network, each epoch goes through all of the datapoints
        for i in range(num_epochs):
            network_output = example2_NN.train(input_, output)
            
            
        ################# Print Results#######################################
        
        print("##################CUSTOM NETWORK RESULTS######################")
        print(f"Network output: {network_output}")
        
        # Printing the weights going Layers left to right & Neurons top to bottom
        for i, lays in enumerate(example2_NN.layers):

            # Print neuron weights depending on the type of layer they are in            
            if lays.id == "CL":
                
                for j in range(lays.numKernels):
                    # print weights of 1 neuron from a kernel channel
                    print()
                    print(f"Convolutional Layer {i}, Kernel {j} Weights")
                    print(np.reshape(lays.neurons[j, 0, 0].params[:-1], (lays.inputDim[0], lays.kernelSize, lays.kernelSize)))
                    print()
                    print(f"Convolutional Layer {i}, Kernel {j} Bias")
                    print(lays.neurons[j, 0, 0].params[-1])
                    print()
            elif lays.id == "FCL":
                
                for j in range(len(lays.neurons)):
                    print()
                    print(f"Fully Connected Layer {i} Neuron {j} Weights")
                    print(lays.neurons[j].params[:-1])
                    print()
                    print(f"Fully Connected Layer {i} Neuron {j} Bias")
                    print(lays.neurons[j].params[-1])
            else:
                pass

        # TensorFlow Results
        print("####################TENSORFLOW RESULTS#########################")
        example2()

    elif (sys.argv[1] == 'example3'):
        from tensorflowtest_example3 import example3
        from parameters import generateExample3

        num_epochs = 1
        
        
        ################# Get weights, biases, input, output ##################
        
        
        l1k1, l1k2, l1b1, l1b2, l3, l3b, input_, output = generateExample3()
        
        # reshape some generated values
        input_ = input_.reshape((1, 8, 8))
        l3_params = np.concatenate((l3, l3b.reshape(1,1)), axis=1)
        
        
        ################# Construct Network and Run ###########################
        
        
        # self, inputSize, loss, lr
        example2_NN = NeuralNetwork((1, 8, 8), "square error", 100.0)
        
        # self, layerType, activation, numOfNeurons=None, numKernels=None, kernelSize=None, weights=None
        example2_NN.addLayer("CL", activation="logistic", numKernels=2, kernelSize=3, weights=np.array([[l1k1], [l1k2]]), biases=np.array([l1b1, l1b2]))
        example2_NN.addLayer("MPL", pool_size=2)
        example2_NN.addLayer("FL")
        example2_NN.addLayer("FCL", "logistic", numOfNeurons=1, weights=l3_params)
        
        # Train the network
        for i in range(num_epochs):
            network_output = example2_NN.train(input_, output)
        
        
        ################# Print Results#######################################
        print("##################CUSTOM NETWORK RESULTS######################")
        
        print(f"Network output: {network_output}")
        
        # Printing the weights going Layers left to right & Neurons top to bottom
        for i, lays in enumerate(example2_NN.layers):

            # Print neuron weights depending on the type of layer they are in            
            if lays.id == "CL":
                
                for j in range(lays.numKernels):
                    # print weights of 1 neuron from a kernel channel
                    print()
                    print(f"Convolutional Layer {i}, Kernel {j} Weights")
                    print(np.reshape(lays.neurons[j, 0, 0].params[:-1], (lays.inputDim[0], lays.kernelSize, lays.kernelSize)))
                    print()
                    print(f"Convolutional Layer {i}, Kernel {j} Bias")
                    print(lays.neurons[j, 0, 0].params[-1])
                    print()
            elif lays.id == "FCL":
                
                for j in range(len(lays.neurons)):
                    print()
                    print(f"Fully Connected Layer {i} Neuron {j} Weights")
                    print(lays.neurons[j].params[:-1])
                    print()
                    print(f"Fully Connected Layer {i} Neuron {j} Bias")
                    print(lays.neurons[j].params[-1])
            else:
                pass
 
        # TensorFlow Results
        print("####################TENSORFLOW RESULTS#########################")
        example3()
        