import numpy as np

class LinearActivation:
    @staticmethod
    def forward(x):
        return x
    
    @staticmethod
    def backward(dA, x):
        return dA

class ReLU:
    @staticmethod
    def forward(x):
        return np.maximum(0, x)
    
    @staticmethod
    def backward(dA, x):
        dZ = np.array(dA, copy=True)
        dZ[x <= 0] = 0
        return dZ

class Sigmoid:
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def backward(dA, x):
        s = 1 / (1 + np.exp(-x))
        return dA * s * (1 - s)

class Tanh:
    @staticmethod
    def forward(x):
        return np.tanh(x)
    
    @staticmethod
    def backward(dA, x):
        return dA * (1 - np.tanh(x)**2)

class Softmax:
    @staticmethod
    def forward(x):
        exps = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)
    
    # Softmax backward is usually combined with cross-entropy loss
    @staticmethod
    def backward(dA, x):
        pass  # Placeholder, handled in loss function


class DeepNeuralNetwork:
    def __init__(self, layer_dims, activations):
        self.parameters = {}
        self.activations = activations
        self.L = len(layer_dims)
        
        # Initialize parameters
        for l in range(1, self.L):
            self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    def forward_propagation(self, X):
        cache = {'A0': X}
        
        for l in range(1, self.L):
            A_prev = cache['A' + str(l-1)]
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            activation_func = globals()[self.activations[l-1]]
            A = activation_func.forward(Z)
            cache['A' + str(l)] = A
            cache['Z' + str(l)] = Z
            
        return A, cache
    
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m
        cost = np.squeeze(cost)
        return cost

    def backward_propagation(parameters, cache, X, Y):

        gradients = {}

        dAL = "Error at output layer"
        gradients['dW2'] = "Gradient of loss with respect to W2"
        gradients['db2'] = "Gradient of loss with respect to b2"
        

        dA1 = "Backpropagated error to hidden layer" 
        gradients['dW1'] = "Gradient of loss with respect to W1"
        gradients['db1'] = "Gradient of loss with respect to b1"
        
        return gradients