import numpy as np
from sigmoid import sigmoid
from scipy import optimize
from CharRNNstart import cost_charRNNstart as cr
from CharRNNstart.encode import encode, decode
#X = np.matrix('[0;0;0;0;1;1;1;0;1;1;1;0;0;0;0;0;1;1;1;0;1;1;1;0;0;0;0;0;1;1;1;0;1;1;1;0]')# 12x1
#X = np.matrix('0,0,1,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,1,0,0; 1,0,0,0,0,0,0,0,0,0,0; 0,0,1,0,0,0,0,0,0,0,0')
#X = encode('tod sat on the earth and ate a ton o horses then he had an idea')
X = encode('tod sat on earth')
numIn = 11
numHid = 70
numOut = 11
numInTot = numIn + numHid + 1
theta1 = np.matrix( 1 * np.sqrt ( 6 / ( numIn + numHid) ) * np.random.randn( numIn + numHid + 1, numHid ) )
theta2 = np.matrix( 1 * np.sqrt ( 6 / ( numHid + numOut ) ) * np.random.randn( numHid + 1, numOut ) )
thetaVec = np.concatenate((theta1.flatten(), theta2.flatten()), axis=1)

opt = optimize.fmin_tnc(cr.costRNN, thetaVec, args=((X)), maxfun=2000)
print('Max fun calls: %s' % (opt[2]))
optTheta = np.array(opt[0])
theta1 = optTheta[0:(numInTot * numHid)].reshape(numInTot, numHid)
theta2 = optTheta[(numInTot * numHid):].reshape(numHid+1, numOut)


def runForward(Xt, theta1, theta2, steps=None):
	data = Xt
	m = X.shape[0]
	if not steps:
		steps = (m - 1)
	#forward propagation
	hid_last = np.zeros((numHid, 1)) #context units
	results = np.zeros((steps, numOut))
	for j in range(steps):#for every input element
		if j >= Xt.shape[0]:
			data = np.round(results[j, :].reshape(1, 11))
		else:
			data = Xt[j, :].reshape(1, 11)
		context = hid_last
		x_context = np.concatenate((data, context.T), axis=1)
		a1 = np.matrix(np.concatenate((x_context, np.matrix('[1]')), axis=1)).T#add bias, context units to input layer
		z2 = theta1.T * a1 #2x1
		a2 = np.concatenate((sigmoid(z2), np.matrix('[1]'))) #add bias, output hidden layer
		hid_last = a2[0:-1, 0]
		z3 = theta2.T * a2 #1x1
		a3 = sigmoid(z3)
		results[j, :] = a3.reshape(numOut,)
	return results

#Xt = np.matrix('0,0,1,0,0,0,0,0,0,0,0')
Xt = encode('ea')
ans = np.round(runForward(Xt, theta1, theta2, 7))
ans2 = decode(ans)
print(ans2)
