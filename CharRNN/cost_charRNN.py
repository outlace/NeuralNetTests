import numpy as np
from sigmoid import sigmoid
def costRNN(thetaVec, *args):
	X = np.matrix(np.array(args))
	numIn, numHid, numOut = 4, 10, 4
	numInTot = numIn + numHid + 1
	theta1 = thetaVec[0:(numInTot * numHid)].reshape(numInTot, numHid)
	theta2 = thetaVec[(numInTot * numHid):].reshape(numHid+1, numOut)
	theta1_grad = np.zeros((numInTot, numHid))
	theta2_grad = np.zeros((numHid + 1, numOut))
	hid_last = np.zeros((numHid, 1))
	m = X.shape[0]
	J = 0
	results = np.zeros((m, numOut))
	for j in range(m-1): #for every training element
		#y = X[j+1,:] #expected output, the next element in the sequence
		y = X[j+1, :]
		context = hid_last
		x_context = np.concatenate((X[j, :], context.T), axis=1)
		a1 = np.matrix(np.concatenate((x_context, np.matrix('[1]')), axis=1)).T#add bias, context units to input layer; 3x1
		z2 = theta1.T * a1; #2x1
		a2 = np.concatenate((sigmoid(z2), np.matrix('[1]'))) #add bias, output hidden layer; 3x1
		hid_last = a2[0:-1, 0];
		z3 = theta2.T * a2 #1x1
		a3 = sigmoid(z3)
		results[j, :] = a3.reshape(numOut,)
		#Backpropagation:::
		#calculate delta errors
		d3 = (a3.T - y)
		d2 = np.multiply((theta2 * d3.T), np.multiply(a2, (1 - a2)))
		#accumulate gradients
		theta1_grad = theta1_grad + (d2[0:numHid, :] * a1.T).T
		theta2_grad = theta2_grad + (a2 * d3)
	for n in range(m-1):
		a3n = results[n, :].T.reshape(numOut, 1)
		yn = X[n+1, :].T
		J = J + (-yn.T * np.log(a3n) - (1-yn).T * np.log(1-a3n))
	J = (1/m) * J
	grad = np.concatenate((theta1_grad.flatten(), theta2_grad.flatten()), axis=1)
	return J, grad
