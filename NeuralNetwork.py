import numpy as np


class NeuralNetwork:

    # initialise the neural network
    def __init__(self, layers, learningrate=0.001, activation='sigmoid'):
        self.input_nodes = layers[0]
        self.hidden_nodes = np.array(layers[1:-1]) + 1 # +1 for bias/
        self.output_nodes = layers[-1]
        self.n_layer = len(layers)             # number of layers
        self.n_hidden_layer = len(layers) - 2  # 1 for input, 1 for output layer
        self.lr = learningrate
        self.activation = activation

        self.weights = []

        for i in range(self.n_layer - 1):
            if i ==0 :
                w = (np.random.RandomState(seed=721).rand(self.hidden_nodes[0], self.input_nodes ) - 0.5)
            elif (i == self.n_layer - 2):
                w = (np.random.RandomState(seed=721).rand(self.output_nodes, self.hidden_nodes[-1]) - 0.5)
            else:
                w =  (np.random.RandomState(seed=721).rand(self.hidden_nodes[i], self.hidden_nodes[i-1]) - 0.5)

            self.weights.append(w)

        self.bias = []
        for i in range(self.n_layer - 1):
             b = np.array(np.ones(layers[i+1]), ndmin=2).T -0.99
             self.bias.append(b)



    # train the neural network
    def train(self, X_train, y_train, epoch):
        for e in range(epoch):
            for i in range(len(X_train)):
                # convert inputs list to 2d array
                inputs = np.array(X_train[i,:], ndmin=2).T
                target = np.zeros(self.output_nodes)
                target[int(y_train[i])] = 1
                targets = np.array(target, ndmin=2).T

                layer_outputs = []

                for i in range(self.n_layer - 1):
                    if i == 0:
                        self.activation_flag = 'forward'
                        ho = self.activation_function(np.dot(self.weights[0], inputs))
                    elif i == self.n_layer - 2:
                        self.activation_flag = 'output_layer'
                        ho = self.activation_function(np.dot(self.weights[i], layer_outputs[i - 1]))
                    else:
                        self.activation_flag = 'forward'
                        ho = self.activation_function(np.dot(self.weights[i], layer_outputs[i - 1]))
                    layer_outputs.append(ho)

                final_outputs = layer_outputs[-1]


                # output layer error is the (target - actual)
                output_errors = targets - final_outputs

                last_hidden_errors = 0
                for i in range(len(self.weights) - 1, -1, -1):
                    self.activation_flag = 'back'
                    if i==len(self.weights)- 1:
                        self.weights[i] += self.lr * np.dot(output_errors * self.activation_function(final_outputs),
                                                            np.transpose(layer_outputs[i-1]))
                        last_hidden_errors = np.dot(self.weights[i].T, output_errors)

                    elif i == 0:
                        self.weights[i] += self.lr * np.dot((last_hidden_errors * self.activation_function(layer_outputs[i])),
                                                            np.transpose(inputs))
                    else:
                        self.weights[i] += self.lr * np.dot((last_hidden_errors * self.activation_function(layer_outputs[i])),
                                                np.transpose(layer_outputs[i-1]))
                        last_hidden_errors = np.dot(self.weights[i].T, last_hidden_errors)



    def predict(self, inputs_list):

        self.activation_flag = 'forward'
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        layer_outputs = []

        for i in range(self.n_layer - 1):
            if i == 0:
                self.activation_flag = 'forward'
                ho = self.activation_function(np.dot(self.weights[0], inputs))
            elif i == self.n_layer - 2:
                self.activation_flag = 'output_layer'
                ho = self.activation_function(np.dot(self.weights[i], layer_outputs[i - 1]))
            else:
                self.activation_flag = 'forward'
                ho = self.activation_function(np.dot(self.weights[i], layer_outputs[i - 1]))
            layer_outputs.append(ho)

        final_outputs = layer_outputs[-1]

        return final_outputs

    def activation_function(self, inputs):
        if self.activation == 'sigmoid':
            if self.activation_flag == 'forward':
                result = self.sigmoid(inputs)
            elif self.activation_flag == 'output_layer':
                result = self.softmax(inputs)
            else:
                result =self.sigmoid_derivative(inputs)
        else:
            if self.activation_flag == 'forward':
                result =self.relu(inputs)
            else:
                result =self.relu_derivative(inputs)
        return result


    def sigmoid(self, z):
        # Sigmoid function
        return (1.0 / (1.0 + np.exp(-z)))


    def relu(self, z):
        # Rectified Linear function
        if np.isscalar(z):
            result = np.max((z, 0))
        else:
            zero_aux = np.zeros(z.shape)
            meta_z = np.stack((z, zero_aux), axis=-1)
            result = np.max(meta_z, axis=-1)
        return result


    def sigmoid_derivative(self, y):
        # Derivative for Sigmoid function
        # Assume y is already sigmoided
        result = y * (1 - y)
        return result


    def relu_derivative(self, z):
        # Derivative for Rectified Linear function
        result = 1 * (z > 0)
        return result

    def softmax(self, x):
        e_x = np.exp(x)
        return e_x / np.sum(e_x)