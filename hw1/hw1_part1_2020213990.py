import numpy as np
import time
import math

## Network architecture
NUM_INPUT = 784  # Number of input neurons
NUM_OUTPUT = 10  # Number of output neurons

## Hyperparameters
NUM_HIDDEN = 50
LEARNING_RATE = 0.05
BATCH_SIZE = 5
NUM_EPOCH = 100

## Hidden value
Y_hat = None
H1 = None
Z1 = None

print("NUM_HIDDEN: ", NUM_HIDDEN)
print("LEARNING_RATE: ", LEARNING_RATE)
print("BATCH_SIZE: ", BATCH_SIZE)
print("NUM_EPOCH: ", NUM_EPOCH)


# Load the images and labels from a specified dataset (train or test).
def loadData(which):
    images = np.load("./data/mnist_{}_images.npy".format(which))
    labels = np.load("./data/mnist_{}_labels.npy".format(which))
    return images, labels


## 1. Forward Propagation
# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute the cross-entropy (CE) loss.
def fCE(X, Y, W1, b1, W2, b2):
    global Y_hat, H1, Z1, BATCH_SIZE
    Z1 = np.dot(X, W1) + b1
    H1 = np.maximum(0, Z1)
    z2 = np.dot(H1, W2) + b2
    S = np.exp(z2 - np.max(z2))
    Y_hat = S / (np.sum(S, axis=1).reshape(BATCH_SIZE, 1))
    loss = -np.sum(np.log(Y_hat) * Y) / BATCH_SIZE
    return loss


def predict(X, Y, W1, b1, W2, b2):
    total_ac = 0
    data_size = X.shape[0]
    z1 = np.dot(X, W1) + b1
    h1 = np.maximum(0, z1)
    z2 = np.dot(h1, W2) + b2
    exp_z = np.exp(z2 - np.max(z2))
    y_hat = exp_z / (np.sum(exp_z, axis=1).reshape(data_size, 1))
    for i in range(data_size):
        pred = np.argmax(y_hat[i])
        gt = np.argmax(Y[i])
        if pred == gt:
            total_ac += 1
    return total_ac / data_size


## 2. Backward Propagation
# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute the gradient of fCE. 
def gradCE(X, Y, W2):
    global Y_hat, H1, Z1, BATCH_SIZE
    delta_W2 = np.dot(np.transpose(H1), (Y_hat - Y)) / BATCH_SIZE
    delta_b2 = np.sum((Y_hat - Y), axis=0) / BATCH_SIZE
    sgn_z = np.maximum(0, np.sign(Z1))
    delta_W1 = np.dot(np.transpose(X), np.dot((Y_hat - Y), np.transpose(W2)) * sgn_z) / BATCH_SIZE
    delta_b1 = np.sum(np.dot((Y_hat - Y), np.transpose(W2)) * sgn_z, axis=0) / BATCH_SIZE
    return delta_W1, delta_b1, delta_W2, delta_b2


def learning_rate_cosine_decay(current_step, alpha=0.01):
    global LEARNING_RATE, NUM_EPOCH
    alpha = alpha / LEARNING_RATE
    current_step = min(current_step, NUM_EPOCH)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / NUM_EPOCH))
    decayed = (1 - alpha) * cosine_decay + alpha
    decayed_learning_rate = LEARNING_RATE * decayed
    return decayed_learning_rate


## 3. Parameter Update
# Given training and testing datasets, train the NN.
def train(trainX, trainY, testX, testY):
    global NUM_EPOCH, NUM_HIDDEN, NUM_OUTPUT, NUM_INPUT, BATCH_SIZE
    #  Initialize weights randomly
    W1 = np.random.randn(NUM_INPUT, NUM_HIDDEN) * np.sqrt(2 / NUM_INPUT)
    b1 = np.zeros(NUM_HIDDEN)
    W2 = np.random.randn(NUM_HIDDEN, NUM_OUTPUT) * np.sqrt(2 / NUM_HIDDEN)
    b2 = np.zeros(NUM_OUTPUT)

    data_size = trainX.shape[0]
    for epoch in range(NUM_EPOCH):
        learning_rate = learning_rate_cosine_decay(epoch)
        loss_sum = 0
        count = 0
        for i in range(0, data_size, BATCH_SIZE):
            batchX = trainX[i:i + BATCH_SIZE, :]
            batchY = trainY[i:i + BATCH_SIZE, :]
            loss_sum += fCE(batchX, batchY, W1, b1, W2, b2)
            delta_W1, delta_b1, delta_W2, delta_b2 = gradCE(batchX, batchY, W2)
            W1 = W1 - learning_rate * delta_W1
            b1 = b1 - learning_rate * delta_b1
            W2 = W2 - learning_rate * delta_W2
            b2 = b2 - learning_rate * delta_b2
            count += 1
        train_accuracy = predict(trainX, trainY, W1, b1, W2, b2)
        test_accuracy = predict(testX, testY, W1, b1, W2, b2)
        print('epoch:{}, train_accuracy:{}, test_accuracy:{}, loss:{}, learning_rate:{}'.format(
            epoch, train_accuracy, test_accuracy, loss_sum / count, learning_rate))


if __name__ == "__main__":
    # Load data
    start_time = time.time()
    trainX, trainY = loadData("train")
    testX, testY = loadData("test")

    print("len(trainX): ", len(trainX))
    print("len(testX): ", len(testX))

    # # Train the network and report the accuracy on the training and test set.
    train(trainX, trainY, testX, testY)
