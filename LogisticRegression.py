import sys
import numpy as np
import pandas as pd

# simple logistic regression model meant to classify MNIST dataset images
# input data is expected to be matrix of images unrolled to vectors
# labels are expected to be matrix of vectors with 1 in proper position and 0 in others


#function to load the dataset from csv file
def dataloader(file_name):
    dataset = np.array(pd.read_csv(file_name))
    setlabels = dataset[:, 0]
    setimages = dataset[:, 1:]
    return setimages, setlabels

#function to set the labels matrix
def setlabels(labels):
    labels_formatted = np.zeros((10, np.shape(labels)[0]))
    for i in range(np.shape(labels)[0]):
        labels_formatted[labels[i], i] = 1
    return labels_formatted


# function to calculate sigmoid output layer
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# calculating gradient vector for backprop
def GradientCalc(input, output, labels,theta):
    ex_num = np.shape(output)[1]
    theta0=(1 / ex_num) * np.dot(input[0,:], (output - labels).transpose())
    #adding regularization term
    reg_value=0.0005
    theta=(1 / ex_num) * np.dot(input, (output - labels).transpose())+ (reg_value / (2 * ex_num)) * theta * theta
    theta[0, :]=theta0
    return theta


class LogisticRegressionModel():
    def __init__(self, inp_size, out_size):
        self.inp_size = inp_size
        self.out_size = out_size
        # random weight initialization from uniform distribution and adding bias term
        self.theta = np.random.random((inp_size + 1, out_size))

    def feedforward(self, input):
        # calculating row vector of output layer
        out = np.dot(input.transpose(), self.theta)
        out = out.transpose()
        return sigmoid(out)

    # backpropping calculated gradient
    def backprop(self, input, output, labels, lr_rate,theta):
        self.theta = self.theta - lr_rate * GradientCalc(input, output, labels,theta)

    #main gradient descent handling function
    def start(self, input, labels, epochs, lr_rate, batch_size,testset,testlabels):
        input = np.vstack((np.ones((1, np.shape(input)[1])), input))
        testset = np.vstack((np.ones((1, np.shape(testset)[1])), testset))
        for i in range(epochs):
            hits = 0
            print("EPOCH: ", i + 1)
            # flag indicating end of the file
            flag = 0
            # variable storing actual position
            temp = 0
            while True:
                #checking if batch index wont exceed input array length
                if (np.shape(input)[1]-1-temp>batch_size):
                    # cutting batch from input matrix
                    batch=input[:, temp:temp+batch_size]
                    batchlabels=labels[:, temp:temp+batch_size]
                    temp+=batch_size
                else:
                    batch=input[:,temp:(np.shape(input)[1]-1)]
                    batchlabels = labels[:, temp:(np.shape(input)[1]-1)]
                    flag=1

                # performing GD
                out=self.feedforward(batch)
                self.backprop(batch, out, batchlabels, lr_rate,self.theta)
                if flag==1:
                    break

            # testing accuracy
            for k in range(len(testset)):
                a = testset[:, k]
                b = self.feedforward(a)
                if (np.argmax(b) == np.argmax(testlabels[:, k])):
                    hits += 1

            print("accuracy= ", 100*(hits / len(testset)))

#loading the set
trainset, trainlab = dataloader("mnist_train.csv")
testset, testlab = dataloader("mnist_test.csv")

# setting labels
trainlabels = setlabels(trainlab)
testlabels = setlabels(testlab)

#transposing and normalizing input images
trainset = trainset.transpose()
trainset = trainset/255.0

testset=testset.transpose()
testset=testset/255.0

#setting model parameters
batch_size = 10
lr_rate = 0.075
epochs = 20
model = LogisticRegressionModel(np.shape(trainset)[0], 10)
model.start(trainset, trainlabels, epochs, lr_rate, batch_size, testset, testlabels)


#author found given parameters to work quite well giving ~90% accuracy after about 5 epochs
#in further iterations model starts to visibly overfit
