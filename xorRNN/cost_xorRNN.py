#cost_xorRNN.py
import numpy as np
from sigmoid import sigmoid
def costRNN(thetaVec, *args):
	X = args[0]
	Y = args[1]
	numIn, numHid, numOut = 1, 4, 1
	#reconstitute our theta1 and theta2 from the unrolled thetaVec
	theta1 = thetaVec[0:24].reshape(numIn + numHid + 1, numHid)
	theta2 = thetaVec[24:].reshape(numHid + 1, numOut)
	#initialize our gradient vectors
	theta1_grad = np.zeros((numIn + numHid + 1, numHid))
	theta2_grad = np.zeros((numHid + 1, numOut))
	#this will keep track of the output from the hidden layer
	hid_last = np.zeros((numHid, 1))
	m = X.shape[0]
	J = 0 #cost output
	results = np.zeros((m, 1)) #to store the output of the network
	#this is to find the gradients:
	for j in range(m): #for every training element
		#y = X[j+1,:] #expected output, the next element in the sequence
		y = Y[j]
		context = hid_last
		x_context = np.concatenate((X[j], context)) #add the context units to our input layer
		a1 = np.matrix(np.concatenate((x_context, np.matrix('[1]'))))#add bias, context units to input layer; 3x1
		z2 = theta1.T * a1; #2x1
		a2 = np.concatenate((sigmoid(z2), np.matrix('[1]'))) #add bias, output hidden layer; 3x1
		hid_last = a2[0:-1, 0];
		z3 = theta2.T * a2 #1x1
		a3 = sigmoid(z3)
		results[j] = a3
		#Backpropagation:::
		#calculate delta errors
		d3 = (a3 - y)
		d2 = np.multiply((theta2 * d3), np.multiply(a2, (1 - a2)))
		#accumulate gradients
		theta1_grad = theta1_grad + (d2[0:numHid, :] * a1.T).T
		theta2_grad = theta2_grad + (d3 * a2.T).T
	#calculate the network cost
	for n in range(m):
		a3n = results[n].T
		yn = Y[n].T
		J = J + (-yn.T * np.log(a3n) - (1-yn).T * np.log(1-a3n)) #cross-entropy cost function
	J = (1/m) * J
	grad = np.concatenate((theta1_grad.flatten(), theta2_grad.flatten()), axis=1) #unroll our gradients
	return J, grad