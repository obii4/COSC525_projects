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
        # Assuming wtimesdelta is a lx1 vector of w x delta values from the next layer of size lx1 (Need to figure out where this will come from)!!!
        actderiv = self.activationderivative()

        # (eqn 2 on summary page) (scalar)
        prev_delta = np.sum(wtimesdelta)*actderiv
        


        # (partial E / partial w)'s
        partial_derivatives = prev_delta*self.inputs

        #print(f"sum of deltas = {np.sum(wtimesdelta)}")
        #print(f"wtimesdelta = {wtimesdelta}")
        #print(f"actderiv = {actderiv}")
        #print(f"self.inputs = {self.inputs}")
        #print()
        #print(f"prev_delta = {prev_delta}")
        #print(f"partial_derivative = {partial_derivatives}")


        # adding just the prev_delta term since bias isn't multiplied by an input
        self.partialderivatives = np.append(partial_derivatives, prev_delta)
        
        # w * delta vector to be used in the previous layer (wtimesdelta passed back from this neuron)
        # There will be a set of wtimes delta for each neuron in the layer of this neuron which will form a 2D matrix in the
        # FullyCOnnected class whose transpose will be used in the subsequent layer's backprop
        # this problem could also be fixed if we just appended a 1 to the end of our input vector
        prev_wtimesdelta = prev_delta*self.weights
        prev_wtimesdelta[-1] = prev_delta
        #print(f"prev_wtimesdelta {prev_wtimesdelta}")
        return prev_wtimesdelta
    
    #Simply update the weights using the partial derivatives and the learning weight
    def updateweight(self):

        #print(f"self.weights before update = {self.weights}")

        # Weight update calculation (weights and bias) (eqn 4 on summary page)
        self.weights = self.weights - self.lr * self.partialderivatives

        #print(f"self.weights after update = {self.weights}")

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

        # Create array of neurons in the layer
        self.neurons = []
        for i in range(self.numOfNeurons):

            _neuron_i = Neuron(self.activation, self.input_num, self.lr, self.weights[i,:])
            self.neurons.append(_neuron_i)
        
        
    #calculate the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calculate() method)      
    def calculate(self, input):
        
        neurons_outputs = np.zeros(self.numOfNeurons)

        # Calculate neurons' outputs then store their outputs (Do we need to store the neuron classes?)
        for i in range(self.numOfNeurons):

            _output_i = self.neurons[i].calculate(input)

            neurons_outputs[i] = _output_i
        
        return neurons_outputs
        
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        #wtimesdelta is a 2D matrix coming from the previous layer
        

        # 2D Matrix of wtimesdelta weights (each row is for a neuron's delta*w vector from this layer)
        prev_wtimes_delta = np.zeros([self.numOfNeurons, self.input_num+1])

        # Iterates through neurons top to bottom and calculates the wtimesdelta vector for each one
        for i in range(self.numOfNeurons):

            #print()
            #print()
            #print(f"Overall W Times Delta{wtimesdelta}")
            #print(f" length of wtimesdelta => {len(wtimesdelta)}")

            #y.shape[0], y.shape[1]
            #if len(wtimesdelta) == 1:
            #print(wtimesdelta.shape[0])
            #print(wtimesdelta.shape[1])

            
            if wtimesdelta.shape[0] == 1:
                prev_wtimes_delta[i, :] = self.neurons[i].calcpartialderivative(wtimesdelta)
                #print(f"W times Delta 1st => {wtimesdelta[:, i]}")
            else:
                #print(f"W times Delta 2nd => {wtimesdelta[:, i]}")
                #print()
                #print()
                # Calculates the w*delta vector for the ith neuron
                prev_wtimes_delta[i, :] = self.neurons[i].calcpartialderivative(wtimesdelta[:, i])

            #print(self.neurons[i].weights)

            #update the ith neuron
            #print(self.weights)
            self.neurons[i].updateweight()

            #print(self.neurons[i].weights)

        return prev_wtimes_delta


#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
         
         # Number of Layers excluding the input nodes!!
         self.numOfLayers = numOfLayers
         self.numOfNeurons = numOfNeurons
         self.inputSize = inputSize
         self.activation = activation
         self.loss = loss
         self.lr = lr
         
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
    

         # Create the layers of the network
         self.layers = []
         for i in range(self.numOfLayers):

             if i == 0:
                 # First layer uses the input given by the user
                 _layer_i = FullyConnected(self.numOfNeurons[i], self.activation[i], self.inputSize, self.lr, self.weights[i])
            
             else:
                 # Subsequent layers use the output of previous layers as inputs
                 _layer_i = FullyConnected(self.numOfNeurons[i], self.activation[i], self.numOfNeurons[i-1], self.lr, self.weights[i])

             self.layers.append(_layer_i)

    #Given an input, calculate the output (using the layers calculate() method)
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
        #print(loss_calcs)
        #print(f"len of y = {len(y)}")
        if self.loss == "square error":

            for i in range(len(y)):
                #print(yp[i])
                #print(y[i])
                loss_calcs[i] = (1/len(y)) * np.square(np.subtract(y[i], yp[i]))
                #loss_calcs = (1/len(y)) * np.sum((y[i] - yp[i]) ** 2)

            #print(f"loss_calcs = {loss_calcs}")


        elif self.loss == "binary cross entropy":
            for i in range(len(y)):
                loss_calcs[i] = (1 / len(yp)) * -((y[i] * math.log(yp[i])) + (1 - y[i]) * math.log(1 - yp[i]))

        else:
            print("Unrecognized loss function.")

            return loss_calc_sum
    
        #print(f"loss_calcs = {loss_calcs}")
        loss_calc_sum = np.sum(loss_calcs)
        return loss_calc_sum
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self, yp, y):

        #print(f"yp = {yp}")
        #print(f"y = {y}")


        #loss_derivs = []

        #print(f"NEW SHAPE {y.shape}")
        
        
        
        if sys.argv[1]=='example':
            loss_derivs = np.zeros((y.shape[0], y.shape[1]))
        else:
            loss_derivs = np.zeros(len(y))
        
        
        if self.loss == "square error":
            
            print(len(yp))
            for i in range(len(y)):
                #loss_derivs.append((2/len(yp)*(yp[i] - y[i]))) #loss_deri = (1 / len(y)) * np.sum(2 * (y[i] - yp[i]))
                #print(f"yp[i] = {yp[i]}")
                #print(f"y[i] = {y[i]}")
                #loss_derivs[i] = ((2/len(yp))*(yp[i] - y[i]))
                
                loss_derivs[i] = -(y[i] - yp[i])
                #print(f"loss derivs!!!! = {loss_derivs}")
        elif self.loss == "binary cross entropy":

            for i in range(len(y)):
                loss_derivs.append( -(y[i] / yp[i]) + ((1 - y[i]) / (1 - yp[i])))

        else:
            print("Unrecognized loss function.")

        #return np.array(loss_derivs)

        #print(f"loss_derivs = {loss_derivs}")
        
        
        
        return loss_derivs
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):

        yp = self.calculate(x)
        
        # print output test
        #print(yp.shape)

        loss_deriv = self.lossderiv(yp, y)
        #print(f"loss deriv passed to calcwdeltas = {loss_deriv}")
        next_wdeltas = []

        count = 0
        #for i in range(len(self.layers)):
        for i, e in reversed(list(enumerate(self.layers))):
            #print(count)
            count +=1
            #print(f"i = {i}")
            if i == len(self.layers) - 1:
                # Get wdeltas differently for the last layer
                #print(f"i = {i}")

                #print("Last layer calc")
                #print(f"Loss Derivative is:{loss_deriv}")
                next_wdeltas.append(self.layers[i].calcwdeltas(loss_deriv))
                #next_wdeltas.append(self.layers[i].calcwdeltas(np.array([[0.74136507, -0.217071535]])))
                
                #print(f" loss_deriv passed as a delta = {loss_deriv}")


                #print(self.layers[i].calcwdeltas(loss_deriv))

            else:
                # iterate through the rest of the layers normally
                #next_wdeltas.append(self.layers[i].calcwdeltas(next_wdeltas[i-1]))
                #print(i)

                #print("other layers calc")

                next_wdeltas.append(self.layers[i].calcwdeltas(next_wdeltas[len(self.layers) - (i + 2)]))

                #print(self.layers[i].calcwdeltas(next_wdeltas[i]))
                #print(f"wdeltas passed to neurons: {next_wdeltas[len(self.layers) - (i + 2)]}")

        #return np.reshape(yp, (len(yp), 1))
        return np.reshape(yp, (1, len(yp)))


if __name__=="__main__":
    if (len(sys.argv)<2):




        # Test Code for Neuron Class





        #print("Test Code for Neuron Class")
        neuron_test = Neuron("logistic", 3, 0.01, np.array([0.2, 0.3, 0.4, 0.5]))
        ##neuron_test = Neuron("logistic", 3, 0.01)
        #print(neuron_test.lr)
        #print(neuron_test.input_num)
        #print(neuron_test.activation)
        #print(neuron_test.weights)

        ## Testing the "activate" method

        #print("Testing of activate")
        act_func_val = neuron_test.activate(5)
        #print(act_func_val)

        ## Testing the "calculate" method

        #print("Testing of calculate")
        calc_vect = neuron_test.calculate(np.array([1, 2, 3]))
        #print(calc_vect)
        #print(neuron_test.inputs)
        #print(neuron_test.output)

        ## Testing the "activationderivative" method

        #print("Testing of activationderivative")
        actderiv_val = neuron_test.activationderivative()
        #print(actderiv_val)

        ## Testing "calcpartialderivative" method

        #print("Testing calcpartialderivative")
        prev_deltas = neuron_test.calcpartialderivative(np.array([1, 1]))
        #print(prev_deltas)

        ## Testing of "updateweights" method

        #print("Testing of updateweights")
        #updated_weights = neuron_test.updateweight()
        #print(updated_weights)





        ### Test Code for Fully-Connected Layer Class





        #self,numOfNeurons, activation, input_num, lr, weights=None
        ##print("Testing the FullyConnected Layer Class")

        #layer_test = FullyConnected(2, "logistic", 3, 0.01, np.array([np.array([0.2, 0.3, 0.4, 0.5]), np.array([0.2, 0.3, 0.4, 0.5])])) # in case we have to make numpy arrays of all lists :/
        #layer_test = FullyConnected(2, "logistic", 3, 0.01, np.array([[0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5]]))
        #layer_test.calculate([[1, 2, 3], [1, 2, 3]])
        #f = layer_test.calculate(np.array([1, 2, 3]))
        #print(layer_test.neurons.lr)

        #g = layer_test.calcwdeltas(np.array([np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1])])) # in case we have to make numpy arrays of all lists :/
        #g = layer_test.calcwdeltas(np.array([[3, 3], [3, 3]]))
        #print(g)





        ### Test Code for NeuralNetwork Class 





        #(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None
        #Test Code for Loss
        a = [0, 0, 1, 1]
        p = [1, 1, 1, 1]
        nn = NeuralNetwork(1, [1], 1, 'linear', 'square error', .5)
        loss = nn.lossderiv(a,p)
        #print(loss)

        

        ## This setup DOES work
        W = np.array([np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
                      np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]),
                      np.array([[0.1, 0.1, 0.1, 0.1, 0.1],[0.1, 0.1, 0.1, 0.1, 0.1]])], dtype=object)

        ### This setup does NOT work
        ##W = np.array([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        #              #[[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
        #              #[[0.1, 0.1, 0.1, 0.1, 0.1],[0.1, 0.1, 0.1, 0.1, 0.1]]])
        
        ### 2 x 4 x 2 network size

        ## params: self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None
        #network_test = NeuralNetwork(3, np.array([2, 4, 2]), 2, ["logistic", "logistic", "linear"], "MSE", 0.01, W)

        ## params: self, input
        #network_output = network_test.calculate(np.array([[1, 2]]))
        #print(network_test.layers_output)

        ## 3 x 1 network size

        #W3 = np.array([np.array([[4, 4, 4], [6, 6, 6], [7, 7, 7]]),
        #              np.array([[2, 2, 2, 2]])], dtype=object)

        ## params: self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None
        #network_test = NeuralNetwork(2, np.array([3, 1]), 2, ["linear", "linear"], "square error", 0.01, W3)

        ## params: self, input
        #network_output = network_test.calculate(np.array([[3, 5]]))
        ##print(network_test.layers_output)

        #x = np.array([20, 40])
        #y = np.array([[0.4]])

        #network_train = network_test.train(x, y)



        ### single perceptron test

        W2 = np.array([np.array([[1, 1, 5]])])
        x = np.array([4, 4])
        y = np.array([[10]])


        # params: self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None
        network_test = NeuralNetwork(1, np.array([1]), 2, ["linear"], "square error", 0.01, W2)

        # params: self, input
        #network_output = network_test.calculate(x)
        #print(network_output)

        network_train = network_test.train(x, y)
        
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([[0.05, 0.1]])
        y=np.array([[0.01], [0.99]])

        # params: self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None
        example_network = NeuralNetwork(2, np.array([2, 2]), 2, ["logistic", "logistic"], "square error", 0.5, w)

        network_output = example_network.train(x, y)

        # outputs of feedforward
        print(network_output)

        print(example_network.calculateloss(network_output, y))

        #layer_1 = example_network.layers[0]

        #neuron_1 = layer_1.neurons[1]
        
        #print(neuron_1.weights)




        # Printing the weights of the network

        # weights going Layers left to right & Neurons top to bottom
        for i, lays in enumerate(example_network.layers):

            for j, neurs in enumerate(lays.neurons):
                print(f"Layer {i + 1} Neuron {j + 1} ")
                print(lays.neurons[j].weights[:])






    elif(sys.argv[1]=='and'):
        print("Output of the single perceptron using AND gate data")
        and_inputs = np.array([[0, 0],
                               [0, 1],
                               [1, 0],
                               [1, 1]])

        #and_outputs = np.array([[[0]],
        #                        [[0]],
        #                        [[0]],
        #                        [[1]]])
        and_outputs = np.array([[0],
                                [0],
                                [0],
                                [1]])

        ### Initiate the Neural Network

        W2 = np.array([np.array([[1, 1, 5]])])
        #x = np.array([4, 4])
        #y = np.array([[10]])

        # params: numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights
        #network_test = NeuralNetwork(1, np.array([1]), 2, ["linear"], "square error", 1, W2)
        network_test = NeuralNetwork(1, np.array([1]), 2, ["logistic"], "square error", 0.1)
        #network_test = NeuralNetwork(1, np.array([1]), 2, ["logistic"], "binary cross entropy", 5)

        x = np.array([[0, 0]])
        network_output = network_test.calculate(x)
        #print(f"network output{network_output}")




        ### Check Feedforward:
        x1 = np.array([[0, 0]])
        x2 = np.array([[0, 1]])
        x3 = np.array([[1, 0]])
        x4 = np.array([[1, 1]])
        
        network_output = network_test.calculate(x1)
        print(f"netowrk output {network_output}")
        network_output = network_test.calculate(x2)
        print(f"netowrk output {network_output}")        
        network_output = network_test.calculate(x3)
        print(f"netowrk output {network_output}")        
        network_output = network_test.calculate(x4)
        print(f"netowrk output {network_output}")





        ### Train the Neural Network

        losses = [10]
        counter = 0

        # Train the network until the loss is essentially zero
        #while losses[-1] > 0.01:
        while counter < 1500:
            counter += 1

            # vector of predictions for each datapoint in the dataset
            yp = np.zeros((len(and_outputs),1))
            
            # train network using whole dataset (1 epoch) and update weights each iteration for the datapoint being used
            for i in range(len(and_outputs)):

                # calculate the network output for a datapoint and train the network for that datapoint
                #print(and_inputs[i])
                #print(and_outputs[i].shape[1])


                network_output = network_test.train(and_inputs[i], and_outputs[i])

                # store the prediction for the datapoint
                yp[i] = network_output
                #print(f" YP IS {yp}")


                #break
            
            # Print the predicted output and the loss for each epoch
            #print()
            #print(f"y_predictions at epoch {counter} = {yp.T}.")
            #print(f"newest loss = {network_test.calculateloss(yp, and_outputs):.6f}.")
            #print()
            
            # Append the loss from the epoch to know when to stop the while loop
            losses.append(network_test.calculateloss(yp, and_outputs))
            #break

        
        losses.pop(0)

        ### Plot the convergence
        
        plt.figure(dpi=150)
                
        #figure formatting
        #plt.xlim(-1.5, 1.5)
        #plt.ylim(-0.5, 1.5)
        plt.title("Synthetic Testing Dataset")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")

        plt.plot(range(len(losses)), losses, label = 'Euclidean DB')
        plt.show()

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
        

        #network_test = NeuralNetwork(1, np.array([1]), 2, ["logistic"], "square error", 0.1)
        network_test = NeuralNetwork(2, np.array([2, 1]), 2, ["logistic", "logistic"], "square error", 0.1)

        
        ### Train the Neural Network

        losses = [10]
        counter = 0

        # Train the network until the loss is essentially zero
        #while losses[-1] > 0.01:
        while counter < 1500:
            counter += 1

            # vector of predictions for each datapoint in the dataset
            yp = np.zeros((len(xor_outputs),1))
            
            # train network using whole dataset (1 epoch) and update weights each iteration for the datapoint being used
            for i in range(len(xor_outputs)):

                # calculate the network output for a datapoint and train the network for that datapoint
                #print(and_inputs[i])
                #print(and_outputs[i].shape[1])


                network_output = network_test.train(xor_inputs[i], xor_outputs[i])

                # store the prediction for the datapoint
                yp[i] = network_output
                #print(f" YP IS {yp}")


                #break
            
            # Print the predicted output and the loss for each epoch
            #print()
            #print(f"y_predictions at epoch {counter} = {yp.T}.")
            #print(f"newest loss = {network_test.calculateloss(yp, and_outputs):.6f}.")
            #print()
            
            # Append the loss from the epoch to know when to stop the while loop
            losses.append(network_test.calculateloss(yp, xor_outputs))
            #break

        
        losses.pop(0)

        ### Plot the convergence

        plt.figure(dpi=150)
                
        #figure formatting
        #plt.xlim(-1.5, 1.5)
        #plt.ylim(-0.5, 1.5)
        plt.title("Synthetic Testing Dataset")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")

        plt.plot(range(len(losses)), losses, label = 'Euclidean DB')
        plt.show()
