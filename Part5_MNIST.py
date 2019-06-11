from mlxtend.data import loadlocal_mnist
from NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

X_train, y_train = loadlocal_mnist(
        images_path='./data/train-images.idx3-ubyte',
        labels_path='./data/train-labels.idx1-ubyte')

X_test, y_test = loadlocal_mnist(
        images_path='./data/t10k-images.idx3-ubyte',
        labels_path='./data/t10k-labels.idx1-ubyte')


# epochs is the number of times the training data set is used for training
epoch = 5

learning_rate = 0.001
my_mlp_accuracy = []
for i in range(8, 64, 8):
    layer = [784, i, 10]
    n = NeuralNetwork(layer, learning_rate)

    for k in range(len(X_train)):
        X_train[i] = np.ndarray.flatten(np.asfarray(X_train[k]))

    n.train(X_train, y_train, epoch=epoch)

    scorecard = []
    # go through all the records in the test data set
    for j in range(len(X_test)):
        # scale and shift the inputs
        image = np.ndarray.flatten(np.asfarray(X_train[j]))
        correct_label = y_train[j]
        # query the network
        outputs = n.predict(image)
        # the index of the highest value corresponds to the label
        label = np.argmax(outputs)
        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)


    # calculate the performance score, the fraction of correct answers
    scorecard_array = np.asarray(scorecard)
    print ("My MLP Accuracy = ", scorecard_array.sum() / scorecard_array.size)
    my_mlp_accuracy.append([i, (scorecard_array.sum() / scorecard_array.size)*100])

sklearn_accuracy = []
for i in range(8, 64, 8):
    mlp = MLPClassifier(hidden_layer_sizes=(i), max_iter=epoch,
                        solver='sgd',  random_state=721, momentum=0,
                        learning_rate_init=learning_rate, batch_size=1, early_stopping=False, activation='logistic')
    mlp.fit(X_train, y_train)
    print("SkLearn Accuracy = {}".format(mlp.score(X_test, y_test)))
    sklearn_accuracy.append([i,mlp.score(X_test, y_test)*100])

plt.plot(np.array(my_mlp_accuracy)[:,0],np.array(my_mlp_accuracy)[:,1], label='My MLP')
plt.scatter(np.array(my_mlp_accuracy)[:, 0], np.array(my_mlp_accuracy)[:, 1])
plt.plot(np.array(sklearn_accuracy)[:,0],np.array(sklearn_accuracy)[:,1], label='SkLearn MLP')
plt.scatter(np.array(sklearn_accuracy)[:, 0], np.array(sklearn_accuracy)[:, 1])
plt.xlabel("Number of hidden unit in layer")
plt.ylabel("Accuracy(%)")
plt.title("Number of hidden unit comparison\nlayer [2, x, 3]   epoch={}   lr={}".format(epoch, learning_rate))
plt.ylim([40,100])
plt.legend()
plt.show()