import numpy as np
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ]).T #3x4 (4=num training examples)
y = np.array([[1,0,0,1]]).T #4x1
theta1 = 2*np.random.random((3,1)) - 1 #3x1
alpha = 0.1 #learning rate
epochs = 10

for j in range(epochs):
    theta1_grad = np.zeros((3,1))
    for x in range(X.shape[1]): #for each training example
        #print("When x = %s\n%s" % (x, X[:,x]))
        a1 = X[:,x] #input layer; 1x3
        a2 = 1/(1+np.exp(-(np.dot(theta1.T, a1)))) #output of layer 1; #1x3 * 3x1 = 1x1
        #print("a2:\n%s\n" % (a2))
        a2_delta = np.multiply((a2 - y[x]), a2*(1-a2)) #1x1 - 1x1
        #print("a2_delta:\n%s\n" % (a2_delta))
        grad1 = (a1 * a2_delta).reshape(3,1)#3x1 * 1x1 = 3x1
        #print("grad1: \n%s\n" % (grad1))
        theta1_grad += grad1.reshape(3,1)
        cost = np.sum((a2.T - y)**2 / 4)
    #print("theta1: \n%s\n" % (theta1))
    #print("theta1_grad: \n%s\n" % (theta1_grad.T))
    theta1 = theta1 - (alpha * theta1_grad)
    if j == 0:
        print("Cost Before GD: " + str(cost))
    if j == (epochs-1):
        print("Cost After GD: " + str(cost))
        #print("theta1: %s" % (theta1))

