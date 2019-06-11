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

"""
Create a color list for scatter plot
red for Iris-setosa, green for Iris-versicolor
and yellow for Iris-virginica
"""
colors = []
for i in range(len(y_train)):
    if y_train[i, 0] == 0:
        colors.append('r')
    elif y_train[i, 0] == 1:
        colors.append('g')
    else:
        colors.append('y')

epoch = 100
learning_rate = 0.01

layers = [2, 16, 16, 16, 16, 3]
n = NeuralNetwork(layers, learning_rate)
n.train(X_train, y_train, epoch=epoch)

score = []
for j in range(len(X_test)):
    predict = n.predict(X_test[j,:])
    if(np.argmax(predict) == y_test[j]):
        score.append(1)
    else:
        score.append(0)

# Find each point on coordinate space
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
h = .01  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

prediction_list = np.c_[xx.ravel(), yy.ravel()]

Z = []
for i in range(len(prediction_list)):
    Z.append(np.argmax(n.predict((prediction_list[i]))))

# Put the result into a color plot
Z = np.array(Z).reshape(xx.shape)

plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

print(Z.shape)
space_color_list = []  # create color list for space

# map each point on coordinate to a color according its label
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):

        if (Z[i][j]) == 0:
            space_color_list.append('#FFDDDD')

        elif (Z[i][j]) == 1:
            space_color_list.append('#DDFFDD')

        else:
            space_color_list.append('#FFFFDD')


# plot each point on space
plt.scatter(xx, yy, marker='o', c=space_color_list, s=3)

# Plot also the training points
plt.scatter(X_train[:, [0]], X_train[:, [1]], marker='o', c=colors)

plt.title('Decision Boundaries (My MLP)\nlayer={}  lr={} epoch={}'.format(layers, learning_rate, epoch))
plt.xlabel('Feature 1')
plt.ylabel('Feature 4')
plt.xlim(X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5)
plt.ylim(X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5)
plt.show()

mlp = MLPClassifier(hidden_layer_sizes=(16,16,16,16), max_iter=epoch,
                    solver='sgd', random_state=721, momentum=0, shuffle=False,
                    learning_rate_init=learning_rate, batch_size=1, early_stopping=False, activation='logistic')
mlp.fit(X_train, y_train)
score = mlp.score(X_test, y_test)
# print("Sklearn Accuracy for layers = [2, {}, 3] ==> {}".format(i, score ))

# Find each point on coordinate space
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
h = .01  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))



Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.reshape(Z, xx.shape)

# plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

print(Z.shape)
space_color_list = []  # create color list for space

# map each point on coordinate to a color according its label
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):

        if (Z[i][j]) == 0:
            space_color_list.append('#FFDDDD')

        elif (Z[i][j]) == 1:
            space_color_list.append('#DDFFDD')

        else:
            space_color_list.append('#FFFFDD')


# plot each point on space
plt.scatter(xx, yy, marker='o', c=space_color_list, s=3)

# Plot also the training points
plt.scatter(X_train[:, [0]], X_train[:, [1]], marker='o', c=colors)

plt.title('Decision Boundaries (SkLearn MLP)\nlayer={}  lr={} epoch={}'.format(layers, learning_rate, epoch))
plt.xlabel('Feature 1')
plt.ylabel('Feature 4')
plt.xlim(X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5)
plt.ylim(X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5)
plt.show()