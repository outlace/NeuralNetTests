import numpy as np
import sigmoid as sd

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
		a2 = np.concatenate((sd.sigmoid(z2), np.matrix('[1]'))) #add bias, output hidden layer
		hid_last = a2[0:-1, 0]
		z3 = theta2.T * a2 #1x1
		a3 = sd.sigmoid(z3)
		results[j] = a3
	return results

def costFunctionRNN(X, theta1, theta2):
	m = X.shape[0]
	J = 0
	#forward propagation
	results = runForward(X, theta1, theta2)

	for n in range(m-1):
		a3n = results[n].T
		yn = X[n+1,:].T
		J = J + (-yn.T * np.log(a3n) - (1-yn).T * np.log(1-a3n))
	return (1/m) * J
'''
theta1 = np.matrix([[-2.22088354,  0.02176577,  0.84477795, -1.94074041],
        [-0.12130562,  0.99752492,  0.52409575, -1.96266681],
        [-1.38975829,  0.0769392 , -0.34500347, -1.08993267],
        [ 1.03377708, -0.37843839,  0.21764661, -0.13544274],
        [-0.21921021,  0.30072887,  0.3274492 ,  2.29589407],
        [-1.5020334 ,  0.47715032, -0.17688044,  0.18599133]])
theta2 = np.matrix([[ 0.09363998],
        [ 0.75976435],
        [ 0.85810382],
        [-1.19553176],
        [-0.43841872]])
''''''
theta1 = np.matrix([[-0.58245296,  2.00180108, -0.40156266,  0.10663207],
        [ 0.57194974,  0.41317721, -0.1572666 , -1.3431595 ],
        [-1.25752607, -0.09952857, -2.76149367,  0.48403485],
        [-0.23268267, -1.34799941, -0.00345387, -0.17830732],
        [-0.88962019, -0.26075391,  0.20606983, -0.04101973],
        [-0.59256745, -0.6935694 , -2.45302098,  0.65226764]])
theta2 = np.matrix([[-0.57855142],
        [ 2.06681618],
        [-0.78477725],
        [-0.79023812],
        [-0.37926865]])
''''''
#X = np.matrix('[0;0;0;0;1;1;1;0;1;1;1;0;0;0;0;0;1;1;1;0;1;1;1;0;0;0;0;0;1;1;1;0;1;1;1;0]')# 12x1
X = np.matrix('[0;0;1;1;0]')
theta1 = np.matrix('[-1.4602,-0.3431,0.8993,-0.8305;-0.5262,2.4998,0.0422,0.6575;-0.6069,0.9094,-1.4545,0.2179;0.3016,-1.4771,-0.9378,-0.7091;-0.1277,0.9335,-0.1378,-0.0775;1.0637,0.8422,2.1801,-0.5987]')
theta2 = np.matrix('[-1.7980;0.2258;1.4898;-0.6946;-0.1696]')
numIn, numHid, numOut = 1, 4, 1
#theta1 = np.matrix( 1 * np.sqrt ( 6 / ( numIn + numHid) ) * np.random.randn( numIn + numHid + 1, numHid ) )
#theta2 = np.matrix( 1 * np.sqrt ( 6 / ( numHid + numOut ) ) * np.random.randn( numHid + 1, numOut ) )
results = runForward(X, theta1, theta2)
print(np.round(results).reshape(1,results.size))
cost = costFunctionRNN(X, theta1, theta2)
print(cost)
'''