import numpy as np
from sigmoid import sigmoid
from scipy import optimize
from xorRNN import cost_xorRNN as cr
X = np.matrix('[0;0;1;1;0]')
Y = np.matrix('[0;0;1;0;1]')
numIn = 1
numHid = 4
numOut = 1
theta1 = np.matrix( 1 * np.sqrt ( 6 / ( numIn + numHid) ) * np.random.randn( numIn + numHid + 1, numHid ) )
theta2 = np.matrix( 1 * np.sqrt ( 6 / ( numHid + numOut ) ) * np.random.randn( numHid + 1, numOut ) )
thetaVec = np.concatenate((theta1.flatten(), theta2.flatten()), axis=1)
opt = optimize.fmin_tnc(cr.costRNN, thetaVec, args=(X, Y), maxfun=2000)
optTheta = np.array(opt[0])
theta1 = optTheta[0:24].reshape(6, 4)
theta2 = optTheta[24:].reshape(5, 1)

def runForward(X, theta1, theta2):
	m = X.shape[0]
	numIn, numHid, numOut = 1, 4, 1
	#forward propagation
	hid_last = np.zeros((numHid, 1)) #context units
	results = np.zeros((m, 1))
	for j in range(m):#for every input element
		context = hid_last
		x_context = np.concatenate((X[j,:], context))
		a1 = np.matrix(np.concatenate((x_context, np.matrix('[1]'))))#add bias, context units to input layer
		z2 = theta1.T * a1 #2x1
		a2 = np.concatenate((sigmoid(z2), np.matrix('[1]'))) #add bias, output hidden layer
		hid_last = a2[0:-1, 0]
		z3 = theta2.T * a2 #1x1
		a3 = sigmoid(z3)
		results[j] = a3
	return results

Xt = np.matrix('[1;0;0;1;1;0]')
print(np.round(runForward(Xt, theta1, theta2)))