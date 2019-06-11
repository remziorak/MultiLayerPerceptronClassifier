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
my_mlp_result = []
for i in np.arange(0.01, 0.9, 0.02):
    learning_rate = i
    layers = [2, 16, 16, 3]
    n = NeuralNetwork(layers, learning_rate)
    n.train(X_train, y_train, epoch=epoch)

    score = []
    for j in range(len(X_train)):
        predict = n.predict(X_train[j,:])
        if(np.argmax(predict) == y_train[j]):
            score.append(1)
        else:
            score.append(0)
    accuracy = (sum(score)/len(score)) * 100
    print('Accuracy for layers = {} lr = {} epoch = {} ==> {}'.format(layers, learning_rate, epoch, accuracy))
    my_mlp_result.append([i, accuracy])


sklearn_result = []
for i in np.arange(0.01, 0.9, 0.02):
    mlp = MLPClassifier(hidden_layer_sizes=(16,16), max_iter=epoch,
                        solver='sgd', random_state=721, momentum=0, shuffle=False,
                        learning_rate_init=i, batch_size=1, early_stopping=False, activation='logistic')
    mlp.fit(X_train, y_train)
    accuracy = mlp.score(X_test, y_test) * 100
    print("Sklearn Accuracy for layers = [2, 16, 16, 3] ==> {}".format(accuracy))
    sklearn_result.append([i, accuracy])


plt.plot(np.array(my_mlp_result)[:,0],np.array(my_mlp_result)[:,1], label='My MLP')
plt.scatter(np.array(my_mlp_result)[:, 0], np.array(my_mlp_result)[:, 1])
plt.plot(np.array(sklearn_result)[:,0],np.array(sklearn_result)[:,1], label='SkLearn MLP')
plt.scatter(np.array(sklearn_result)[:, 0], np.array(sklearn_result)[:, 1])
plt.xlabel("Learning rate")
plt.ylabel("Accuracy(%)")
plt.title("Laerning rate accuracy comparison\nlayer [2, 16, 16, 3]   epoch={}  ".format(epoch))
plt.ylim([20,100])
plt.legend()
plt.show()