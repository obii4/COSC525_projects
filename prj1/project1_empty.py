import numpy as np
import sys
import math

"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""


# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr, weights=None):
        self.activation = activation
        self.input_num = input_num
        self.lr = lr

        # the last value of the weights vector is the bias
        self.weights = weights if weights is not None else np.random.rand(input_num+1)

        #initialize inputs and outputs arrays for saving feedforward values for later use in backprop
        self.inputs = None
        self.output = None
        
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
        # Assuming wtimesdelta is a lx1 vector of w x delta values from the next layer of size lx1 (Need to figure out where this will come from)!!!
        actderiv = self.activationderivative()

        # (eqn 2 on summary page)
        prev_delta = np.sum(actderiv * wtimesdelta)

        return prev_delta
    
    #Simply update the weights using the partial derivatives and the learning weight
    def updateweight(self):
        print(self.weights)

        # Hardcoded wtimesdelta needs to change!!! Conditional here for if the neuron is before the final layer or not!!!
        prev_delta = self.calcpartialderivative(np.array([2, 2, 2]))

        # Partial derivative of Error wrt to weights (eqn 3 on summary page)
        weights_partial_derivatives = prev_delta * self.inputs

        # Weight update calculation (eqn 4 on summary page)
        self.weights[:-1] = self.weights[:-1] - self.lr * weights_partial_derivatives

        return self.weights

        
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initialize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights if weights is not None else np.random.rand(input_num+1) # This may need work? It is 1D right now

        # list of neuron classes created in calculate() (May or may not be useful when coding backprop)
        self.neurons = None
        
    #calculate the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calculate() method)      
    def calculate(self, input):
        
        neurons =[]
        neurons_output = []

        # Create neurons and calculate their outputs then store the neuron classes and their outputs (Do we need to store the neuron classes?)
        for i in range(self.numOfNeurons):

            _neuron_i = Neuron(self.activation, self.input_num, self.lr, self.weights[i,:])
            _output_i = _neuron_i.calculate(input)

            neurons.append(_neuron_i)
            neurons_output.append(_output_i)
        
        # May not need this, see description in attributes
        self.neurons = neurons

        return neurons_output
        
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        print('calcwdeltas') 
           
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
         self.numOfLayers = numOfLayers
         self.numOfNeurons = numOfNeurons
         self.inputSize = inputSize
         self.activation = activation
         self.loss = loss
         self.lr = lr

         # List of layer classes created in calculate method (May or may not be useful when coding backprop)
         self.layers = None

         # initialize layers outputs attribute from calculate method
         self.layers_output = None
         
         # Use given weights else create a 3D matrix of weights 
         if weights is not None:
             self.weights = weights 
         else:
            # Can make this simpler/shorter by appending inputSize to the beginning of the numOfNeurons attribute then conditional isn't needed 
            weights = []
            for i in range(numOfLayers):
                if i == 0:
                    # weights & biases of first layer
                    temp = np.random.rand(numOfNeurons[0], inputSize+1)
                    weights.append(temp)
                else:
                    # weights & biases for all other layers
                    temp = np.random.rand(numOfNeurons[i], numOfNeurons[i-1]+1)
                    weights.append(temp)

            #Which is better to use??
            #self.weights = np.array(weights, dtype=object)
            self.weights = weights
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        
        layers =[]
        layers_output = []
        for i in range(self.numOfLayers):

            if i == 0:
                # First layer uses the input given by the user
                _layer_i = FullyConnected(self.numOfNeurons[i], self.activation[i], self.inputSize, self.lr, self.weights[i])
                _layer_output_i = _layer_i.calculate(input)
            
            else:
                # Subsequent layers use the output of previous layers as inputs
                _layer_i = FullyConnected(self.numOfNeurons[i], self.activation[i], self.inputSize, self.lr, self.weights[i])
                _layer_output_i = _layer_i.calculate(layers_output[i-1])

            # May not need this, see description in attributes
            layers.append(_layer_i)
            
            # Need this for creating the 2D matrix of all NN layer outputs
            layers_output.append(_layer_output_i)

        self.layers = layers
        self.layers_output = layers_output
        
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        print('calculate')
        if self.loss == "square error":
            for i in range(len(y)):
                loss_calc = (1/len(y)) * np.sum((y[i] - yp[i]) ** 2)

                return loss_calc

        elif self.loss == "binary cross entropy":
            for i in range(len(y)):
                loss_calc = np.sum(y[i] * math.log(yp[i]))

                return loss_calc

         else:
            print("Unrecognized loss function.")

            return loss_calc
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self, yp, y):
        print('lossderiv')

        if self.loss == "square error":
            for i in range(len(y)):
                loss_deriv = (1 / len(y)) * np.sum(2 * (y[i] - yp[i]))

                return loss_deriv

        elif self.loss == "binary cross entropy":
            for i in range(len(y)):
                return loss_deriv

        else:
            print("Unrecognized loss function.")

        return loss_deriv
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        print('train')

if __name__=="__main__":
    if (len(sys.argv)<2):



        # Test Code for Neuron Class




        #print("Test Code for Neuron Class")
        #neuron_test = Neuron("logistic", 3, 0.01, [0.2, 0.3, 0.4, 0.5])
        ##neuron_test = Neuron("logistic", 3, 0.01)
        #print(neuron_test.lr)
        #print(neuron_test.input_num)
        #print(neuron_test.activation)
        #print(neuron_test.weights)

        ## Testing the "activate" method

        #print("Testing of activate")
        #act_func_val = neuron_test.activate(5)
        #print(act_func_val)

        ## Testing the "calculate" method

        #print("Testing of calculate")
        #calc_vect = neuron_test.calculate(np.array([1, 2, 3]))
        #print(calc_vect)
        #print(neuron_test.inputs)
        #print(neuron_test.output)

        ## Testing the "activationderivative" method

        #print("Testing of activationderivative")
        #actderiv_val = neuron_test.activationderivative()
        #print(actderiv_val)

        ## Testing "calcpartialderivative" method

        #print("Testing calcpartialderivative")
        #prev_deltas = neuron_test.calcpartialderivative(np.array([1, 1]))
        #print(prev_deltas)

        ## Testing of "updateweights" method

        #print("Testing of updateweights")
        #updated_weights = neuron_test.updateweight()
        #print(updated_weights)




        ### Test Code for Fully-Connected Layer Class




        #self,numOfNeurons, activation, input_num, lr, weights=None
        ##print("Testing the FullyConnected Layer Class")
        ##layer_test = FullyConnected(2, "logistic", 3, 0.01, np.array([0.2, 0.3, 0.4, 0.5]))
        ##layer_test.calculate([[1, 2, 3], [1, 2, 3]])




        ### Test Code for NeuralNetwork Class 

        #Test Code for Loss
        #a = [0, 0, 1, 1]
        #p = [1, 1, 1, 1]
        #nn = NeuralNetwork(1, 1, 1, 'linear', 'square error', .5)
        #loss = nn.lossderiv(a,p)
        #print(loss)

        

        # This setup DOES work
        W = np.array([np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
                      np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]),
                      np.array([[0.1, 0.1, 0.1, 0.1, 0.1],[0.1, 0.1, 0.1, 0.1, 0.1]])], dtype=object)

        ## This setup does NOT work
        #W = np.array([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                      #[[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
                      #[[0.1, 0.1, 0.1, 0.1, 0.1],[0.1, 0.1, 0.1, 0.1, 0.1]]])
        


        ## 2 x 4 x 2 network size

        # params: self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None
        network_test = NeuralNetwork(3, np.array([2, 4, 2]), 2, ["logistic", "logistic", "linear"], "MSE", 0.01, W)

        # params: self, input
        network_output = network_test.calculate(np.array([[1, 2]]))
        print(network_test.layers_output)


        # single perceptron test

        W2 = np.array([np.array([[5, 6, 10]])])
        
        # params: self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None
        network_test = NeuralNetwork(1, np.array([1]), 2, ["linear"], "MSE", 0.01, W2)

        # params: self, input
        network_output = network_test.calculate(np.array([[1, 1]]))
        print(network_test.layers_output)

        
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1])
        y=np.array([0.01,0.99])
        
    elif(sys.argv[1]=='and'):
        print("learn and")
        and_inputs = np.array([[0, 0],
                               [0, 1],
                               [1, 0],
                               [1, 1]])
        and_outputs = np.array([[0],
                                [0],
                                [0],
                                [1]])
        print(and_outputs.shape)
        print(and_inputs.shape)
        
    elif(sys.argv[1]=='xor'):
        print('learn xor')
        xor_inputs = np.array([[0, 0],
                               [0, 1],
                               [1, 0],
                               [1, 1]])
        xor_outputs = np.array([[0],
                                [1],
                                [1],
                                [0]])
        print(xor_outputs.shape)
        print(xor_inputs.shape)
