import numpy as np

class MLP:
    def __init__(self,input_neurons,hidden_neurons,output):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output = output
        self.networks = [self.input_neurons] + self.hidden_neurons + [self.output]

        # weights initialization
        self.weights = []
        for index in range(len(self.networks) - 1):
            weight = np.random.rand(self.networks[index], self.networks[index + 1])
            self.weights.append(weight)

        # make an array to store activations
        self.activations = []
        for index in range(len(self.networks)):
            activation = np.zeros(self.networks[index])
            self.activations.append(activation)

        # make an array to store derivatives
        self.derivatives = []
        for index in range(len(self.networks) - 1):
            derivative = np.random.rand(self.networks[index], self.networks[index + 1])
            self.derivatives.append(derivative)


    def sigmoid(self,x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    def sigmoid_derivative(self,x):
        y = x * (1.0 - x)
        return y

    def MSE(self,target,output):
        y = np.average((target-output)**2)
        return y

    def forward_pass(self,input):
        activations = input
        # store the input to first activations
        self.activations[0] = activations
        # start forward pass
        for i,w in enumerate(self.weights):
            net_input = np.dot(activations,self.weights[i])
            activations = self.sigmoid(net_input)
            self.activations[i+1] = activations

        return activations

    def backpropogation(self,error):
        # dE/dW_i = (y - a_[i+1]) * s'(h_[i+1]) * a_i <- formula
        # dE/dW_[i-1] = (y - a_[i+1]) * s'(h_[i+1]) * W_i * s'(h_[i+1]) * a_i <- formula
        for i in reversed(range(len(self.networks) - 1)):
            # get the activation of next layer
            activation = self.activations[i+1]
            # calculate delta (y - a_[i+1]) * s'(h_[i+1])
            delta = error * self.sigmoid_derivative(activation)
            # reshape into 2d array for computation and transpose
            delta_reshaped = delta.reshape(delta.shape[0],-1).T
            # get current layer activation
            current_activation = self.activations[i]
            # reshape into 2d array for computation
            current_activation_reshaped = current_activation.reshape(current_activation.shape[0],-1)
            # calculate for derivative (a_i * delta)
            self.derivatives[i] = np.dot(current_activation_reshaped,delta_reshaped)
            # calculate error for next layer (y - a_[i+1]) * s'(h_[i+1]) * W_i
            error = np.dot(delta,self.weights[i].T)

    def gradient_descent(self,learning_rate=0.01):
        # update the weights
        for i in range(len(self.weights)):
            weight = self.weights[i]
            derivative = self.derivatives[i]
            weight += derivative * learning_rate

    def train(self,inputs,targets,epochs,learning_rate):
        for i in range(epochs):
            sum_errors = 0
            for j,input in enumerate(inputs):
                target = targets[j]
                output = self.forward_pass(input)
                error = target - output
                self.backpropogation(error)
                self.gradient_descent(learning_rate)
                sum_errors += self.MSE(target,output)
            print('Errors: {} at epoch {}'.format(sum_errors/(j+1),i))


