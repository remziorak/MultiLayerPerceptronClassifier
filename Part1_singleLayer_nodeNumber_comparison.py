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
my_mlp_result = []

for i in range(4, 128, 4):
    layers = [2, i, 3]
    n = NeuralNetwork(layers, learning_rate)
    n.train(X_train, y_train, epoch=epoch)

    score = []
    for j in range(len(X_test)):
        predict = n.predict(X_test[j,:])
        if(np.argmax(predict) == y_test[j]):
            score.append(1)
        else:
            score.append(0)
    my_mlp_result.append([i, (sum(score)/len(score))*100])
    print('Accuracy for layers = {} lr = {} epoch = {} ==> {}'.format(layers, learning_rate, epoch,(sum(score)/len(score))*100))

sklearn_result = []
for i in range(4, 128, 4):
    mlp = MLPClassifier(hidden_layer_sizes=(i), max_iter=epoch,
                        solver='sgd', random_state=721, momentum=0, shuffle=False,
                        learning_rate_init=learning_rate, batch_size=1, early_stopping=False, activation='logistic')
    mlp.fit(X_train, y_train)
    score = mlp.score(X_test, y_test)
    sklearn_result.append([i, score*100])
    print("Sklearn Accuracy for layers = [2, {}, 3] ==> {}".format(i, score ))

plt.plot(np.array(my_mlp_result)[:,0],np.array(my_mlp_result)[:,1], label='My MLP')
plt.scatter(np.array(my_mlp_result)[:, 0], np.array(my_mlp_result)[:, 1])
plt.plot(np.array(sklearn_result)[:,0],np.array(sklearn_result)[:,1], label='SkLearn MLP')
plt.scatter(np.array(sklearn_result)[:, 0], np.array(sklearn_result)[:, 1])
plt.xlabel("Number of hidden unit in layer")
plt.ylabel("Accuracy(%)")
plt.title("Number of hidden unit comparison\nlayer [2, x, 3]   epoch={}   lr={}".format(epoch, learning_rate))
plt.ylim([40,100])
plt.legend()
plt.show()