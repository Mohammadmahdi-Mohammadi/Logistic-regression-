%matplotlib inline
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import numpy as np

#Logistic regression is handy for classification problems
# since it fits an S shaped logistic (or Sigmoid) function
# to the data, squishing the linear equation to an output
# range of 0–1. This convenient range allows logistic regression
# to model the probabilities of a data point belonging to a particular
# class, typically with the decision point

image_pixels = 784
train_data = np.loadtxt("C:/Users/mohammadmahdi/Desktop/mnist_train.csv",delimiter=",")
test_data = np.loadtxt("C:/Users/mohammadmahdi/Desktop/mnist_test.csv", delimiter=",") 

#the logistic model (or logit model) is used to model the probability of
# a certain class or event existing such as pass/fail, win/lose, alive/dead
# or healthy/sick. This can be extended to model several classes of events
# such as determining whether an image contains a cat, dog, lion, etc. Each
# object being detected in the image would be assigned a probability between
# 0 and 1, with a sum of one.

x = 0.99 / 255
trainD = np.asfarray(train_data[:, 1:]) * x + 0.01
testD = np.asfarray(test_data[:, 1:]) * x + 0.01
trainL = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])


lr = np.arange(10)

for temp in range(10):
    one_hot = (lr==temp).astype(np.int)
      
lr = np.arange(10)

trainL_oneHot = (lr==trainL).astype(np.float)
testL_oneHot = (lr==test_labels).astype(np.float)

trainL_oneHot[train_labels_one_hot==0] = 0.01
trainL_oneHot[train_labels_one_hot==1] = 0.99
testL_oneHot[test_labels_one_hot==0] = 0.01
testl_oneHot[test_labels_one_hot==1] = 0.99



#logistic regression by considering a logistic model with given parameters,
# then seeing how the coefficients can be estimated from data. Consider a
# model with two predictors, {\displaystyle x_{1}}x_{1} and {\displaystyle x_{2}}x_{2},
# and one binary (Bernoulli) response variable {\displaystyle Y}Y, which we denote
# {\displaystyle p=P(Y=1)}{\displaystyle p=P(Y=1)}. We assume a linear relationship
# between the predictor variables and the log-odds of the event that {\displaystyle Y=1}Y=1.
# This linear relationship can be written in the following mathematical form (where ℓ is the log-odds,
# {\displaystyle b}b is the base of the logarithm, and {\displaystyle \beta _{i}}\beta _{i} are parameters of the model):


learning_r=0.001
num_iter=2;

def sigmoid(z):
    return 1 / (1 + np.e **(-z))
    
theta = np.zeros((784,1))

for i in range (10):
        for j in range(100):
            X_train_batch = trainD[((j)*600):(j+1)*600,]
            
            X = X_train_batch.reshape(600,784)
            
            y_train_batch=trainL[((j)*600):(j+1)*600,]
            
            for num1 in range (600):
                if(y_train_batch[num1]!=i):
                    y_train_batch[num1] = 0
                else: 
                    y_train_batch[num1] = 1
            
            y = (y_train_batch != 0) * 1
            h = sigmoid(np.dot(X, theta))
            gradient = np.matmul(np.transpose(X), (h - y)) / y.size
            theta = theta- learning_r * gradient

#In a dataset, a training set is implemented to build up a model, while a test (or validation)
# set is to validate the model built. Data points in the training set are excluded from the test (validation) set
            
        X = testD.reshape(10000,784)
        temp = test_labels
        y = (test_labels != 0) * 1
        for num2 in range (600):
                    if(temp[num2]==i):
                        temp[num2] = 1
                    else: 
                        temp[num2] = 0

        y_p=sigmoid(np.matmul(X,theta));


        y_p=(y_p >= 0.5) * 1
        acc=np.sum(y_p==y)/10000
        print(i,"  acc:" ,acc)

