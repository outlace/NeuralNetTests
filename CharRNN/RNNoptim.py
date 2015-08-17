import numpy as np
from sigmoid import sigmoid
from scipy import optimize
from CharRNN import cost_charRNN as cr
#Vocabulary h,e,l,o
#Encoding:   h = [0,0,1,0], e = [0,1,0,0], l = [0,0,0,1], o = [1,0,0,0]
X = np.matrix('0,0,1,0; 0,1,0,0; 0,0,0,1; 0,0,0,1; 1,0,0,0')
numIn, numHid, numOut = 4, 10, 4
numInTot = numIn + numHid + 1
theta1 = np.matrix( 1 * np.sqrt ( 6 / ( numIn + numHid) ) * np.random.randn( numIn + numHid + 1, numHid ) )
theta2 = np.matrix( 1 * np.sqrt ( 6 / ( numHid + numOut ) ) * np.random.randn( numHid + 1, numOut ) )
thetaVec = np.concatenate((theta1.flatten(), theta2.flatten()), axis=1)
opt = optimize.fmin_tnc(cr.costRNN, thetaVec, args=(X), maxfun=5000)
optTheta = np.array(opt[0])
theta1 = optTheta[0:(numInTot * numHid)].reshape(numInTot, numHid)
theta2 = optTheta[(numInTot * numHid):].reshape(numHid+1, numOut)

def runForward(X, theta1, theta2):
	m = X.shape[0]
	#forward propagation
	hid_last = np.zeros((numHid, 1)) #context units
	results = np.zeros((m, numOut))
	for j in range(m):#for every input element
		context = hid_last
		x_context = np.concatenate((X[j,:], context.T), axis=1)
		a1 = np.matrix(np.concatenate((x_context, np.matrix('[1]')), axis=1)).T#add bias, context units to input layer
		z2 = theta1.T * a1
		a2 = np.concatenate((sigmoid(z2), np.matrix('[1]'))) #add bias, output hidden layer
		hid_last = a2[0:-1, 0]
		z3 = theta2.T * a2
		a3 = sigmoid(z3)
		results[j, :] = a3.reshape(numOut,)
	return results
#This spells 'hell' and we expect it to return 'ello' as it predicts the next character for each input
Xt = np.matrix('0,0,1,0; 0,1,0,0; 0,0,0,1; 0,0,0,1')
print(np.round(runForward(Xt, theta1, theta2)))