import numpy as np
import runRNN as rn
from sigmoid import sigmoid
X = np.matrix('[0;0;0;0;1;1;1;0;1;1;1;0;0;0;0;0;1;1;1;0;1;1;1;0;0;0;0;0;1;1;1;0;1;1;1;0]')# 12x1
numIn = 1
numHid = 4
numOut = 1
theta1 = np.matrix( 1 * np.sqrt ( 6 / ( numIn + numHid) ) * np.random.randn( numIn + numHid + 1, numHid ) )
theta2 = np.matrix( 1 * np.sqrt ( 6 / ( numHid + numOut ) ) * np.random.randn( numHid + 1, numOut ) )
theta1_grad = np.zeros((numIn + numHid + 1, numHid))
theta2_grad = np.zeros((numHid + 1, numOut))
epochs = 12000
alpha = 0.006
epsilon = 0.01
#thetaVec = np.concatenate((theta1.ravel(), theta2.ravel()), axis=1)
minErr = 1e-1
hid_last = np.zeros((numHid, 1))
last_change1 = np.zeros((numIn + numHid + 1, numHid))
last_change2 = np.zeros((numHid + 1, numOut))
m = X.shape[0]

err = rn.costFunctionRNN(X, theta1, theta2)
print('Error before training: %s\n' % (err))
for i in range(epochs):
	#forward propagation
	s = 0#np.random.randint(1, (m-1))
	for j in range(s,(m-1)): #for every training element
		y = X[j+1,:] #expected output, the next element in the sequence
		context = hid_last
		x_context = np.concatenate((X[j,:], context))
		a1 = np.matrix(np.concatenate((x_context, np.matrix('[1]'))))#add bias, context units to input layer; 3x1
		z2 = theta1.T * a1; #2x1
		a2 = np.concatenate((sigmoid(z2), np.matrix('[1]'))) #add bias, output hidden layer; 3x1
		hid_last = a2[0:-1, 0];
		z3 = theta2.T * a2 #1x1
		a3 = sigmoid(z3)
		#Backpropagation:::
		#calculate delta errors
		d3 = (a3 - y)
		d2 = np.multiply((theta2 * d3), np.multiply(a2, (1 - a2)))
		#accumulate gradients
		theta1_grad = theta1_grad + (d2[0:numHid, :] * a1.T).T
		theta2_grad = theta2_grad + (d3 * a2.T).T
	#We're using momentum here, need to keep track of the previous weight update
	theta1_change = alpha * (1/m)*theta1_grad + epsilon * last_change1
	theta2_change = alpha * (1/m)*theta2_grad + epsilon * last_change2
	theta1 = theta1 - theta1_change
	theta2 = theta2 - theta2_change
	last_change1 = theta1_change
	last_change2 = theta2_change
	#reset gradients
	theta1_grad = np.zeros((numIn + numHid + 1, numHid))
	theta2_grad = np.zeros((numHid + 1, numOut))

X2 = np.matrix('[0;0;1;1]')
err = rn.costFunctionRNN(X2, theta1, theta2)
print('Error AFTER training: %s\n' % (err))
print('Result:\n')
print(np.round(rn.runForward(X2, theta1, theta2).T))
print('Expected Result:\n')
print(X[1:].T)

