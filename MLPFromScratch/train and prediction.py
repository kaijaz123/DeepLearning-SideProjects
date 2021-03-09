from model import MLP
from random import random
import numpy as np

def train():
    # create a dataset to train a network for the sum operation
    inputs = np.array([[random() / 2 for i in range(2)] for j in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])

    # create a Multilayer Perceptron with one hidden layer and each neuron on the layer
    mlp = MLP(2, [5], 1)

    # train network
    mlp.train(inputs, targets, 50, 0.1)
    return mlp

mlp = train()
inputs = np.array([0.2,0.3])
output = mlp.forward_pass(inputs)
print('Predicted Output for sum operation {} + {} : {}'.format(inputs[0],inputs[1],output))