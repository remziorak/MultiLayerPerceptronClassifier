from NeuralNetwork import NeuralNetwork
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

def split_data(dataset, test_size=0.5):
    """Function shuffle the data points and return splitted sets """
    shuffled_data = np.random.RandomState(seed=721).permutation(dataset)
    train_set = shuffled_data[: int(len(dataset) * (1 - test_size)), :]
    test_set = shuffled_data[int(len(dataset) * (1 - test_size)):, :]
    return train_set, test_set


filename = 'iris_data.txt'

# read data from file
data = pd.read_csv(filename, sep=',', header=None)

# split dataset to train and test set
train_set, test_set = split_data(data, test_size=0.3)

X_train = train_set[:,[0,3]]  # use first and fourth feature of iris dataset
y_train = np.array(train_set[:,4]).reshape(-1,1)
X_test = test_set[:,[0,3]]
y_test = np.array(test_set[:,4]).reshape(-1,1)

epoch = 200
learning_rate = 0.01
my_mlp_relu_result = []
activation = 'relu'
for i in range(4, 128, 4):
    layers = [2, i, 3]
    n = NeuralNetwork(layers, learning_rate, activation=activation)
    n.train(X_train, y_train, epoch=epoch)

    score = []
    for j in range(len(X_test)):
        predict = n.predict(X_test[j,:])
        if(np.argmax(predict) == y_test[j]):
            score.append(1)
        else:
            score.append(0)
    my_mlp_relu_result.append([i, (sum(score)/len(score))*100])

my_mlp_sigmoid_result = []
activation ='sigmoid'
for i in range(4, 128, 4):
    layers = [2, i, 3]
    n = NeuralNetwork(layers, learning_rate, activation=activation)
    n.train(X_train, y_train, epoch=epoch)

    score = []
    for j in range(len(X_test)):
        predict = n.predict(X_test[j,:])
        if(np.argmax(predict) == y_test[j]):
            score.append(1)
        else:
            score.append(0)
    my_mlp_sigmoid_result.append([i, (sum(score)/len(score))*100])


plt.plot(np.array(my_mlp_relu_result)[:,0],np.array(my_mlp_relu_result)[:,1], label='Relu')
plt.scatter(np.array(my_mlp_relu_result)[:, 0], np.array(my_mlp_relu_result)[:, 1])
plt.plot(np.array(my_mlp_sigmoid_result)[:,0],np.array(my_mlp_sigmoid_result)[:,1], label='SkLearn MLP')
plt.scatter(np.array(my_mlp_sigmoid_result)[:, 0], np.array(my_mlp_sigmoid_result)[:, 1])
plt.xlabel("Number of hidden unit in layer")
plt.ylabel("Accuracy(%)")
plt.title("Sigmoid vs Relu\nlayer [2, x, 3]   epoch={}   lr={}".format(epoch, learning_rate))
plt.ylim([40,100])
plt.legend()
plt.show()


epoch = 200
my_mlp_result_relu = []
activation = 'relu'
for i in np.arange(0.001, 0.4, 0.002):
    learning_rate = i
    layers = [2, 16, 16, 3]
    n = NeuralNetwork(layers, learning_rate, activation=activation)
    n.train(X_train, y_train, epoch=epoch)

    score = []
    for j in range(len(X_train)):
        predict = n.predict(X_train[j,:])
        if(np.argmax(predict) == y_train[j]):
            score.append(1)
        else:
            score.append(0)
    accuracy = (sum(score)/len(score)) * 100
    my_mlp_result_relu.append([i, accuracy])

epoch = 200
my_mlp_result_sigmoid = []
activation = 'sigmoid'
for i in np.arange(0.001, 0.4, 0.002):
    learning_rate = i
    layers = [2, 16, 16, 3]
    n = NeuralNetwork(layers, learning_rate, activation=activation)
    n.train(X_train, y_train, epoch=epoch)

    score = []
    for j in range(len(X_train)):
        predict = n.predict(X_train[j,:])
        if(np.argmax(predict) == y_train[j]):
            score.append(1)
        else:
            score.append(0)
    accuracy = (sum(score)/len(score)) * 100
    my_mlp_result_sigmoid.append([i, accuracy])



plt.plot(np.array(my_mlp_result_relu)[:,0],np.array(my_mlp_result_relu)[:,1], label='Relu')
plt.scatter(np.array(my_mlp_result_relu)[:, 0], np.array(my_mlp_result_relu)[:, 1])
plt.plot(np.array(my_mlp_result_sigmoid)[:,0],np.array(my_mlp_result_sigmoid)[:,1], label='Sigmoid')
plt.scatter(np.array(my_mlp_result_sigmoid)[:, 0], np.array(my_mlp_result_sigmoid)[:, 1])
plt.xlabel("Learning rate")
plt.ylabel("Accuracy(%)")
plt.title("Laerning rate accuracy comparison\nlayer [2, 16, 16, 3]   epoch={}  ".format(epoch))
plt.ylim([20,100])
plt.legend()
plt.show()