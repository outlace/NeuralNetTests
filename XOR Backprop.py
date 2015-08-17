import numpy as np
X = np.matrix([ [0,0],[0,1],[1,0],[1,1]]) #4x2 (4=num training examples)
y = np.matrix([[0,1,1,0]]).T #4x1
numIn, numHid, numOut = 2, 3, 1; #setup layers
theta1 = ( 0.5 * np.sqrt ( 6 / ( numIn + numHid) ) * np.random.randn( numIn + 1, numHid ) )
theta2 = ( 0.5 * np.sqrt ( 6 / ( numHid + numOut ) ) * np.random.randn( numHid + 1, numOut ) )
theta1_grad = np.matrix(np.zeros((numIn+1, numHid))) #3x2
theta2_grad = np.matrix(np.zeros((numHid + 1, numOut)))#3x1
alpha = 0.1 #learning rate
epochs = 10000
m = X.shape[0]; #num training examples

def sigmoid(x):
	return np.matrix(1.0 / (1.0 + np.exp(-x)))

def cost(X, y, theta1, theta2):
	J=0
	a1 = np.matrix(np.concatenate((X, np.ones((4,1))), axis=1))
	z2 = np.matrix(a1.dot(theta1))
	a2 = np.matrix(np.concatenate((sigmoid(z2), np.ones((4,1))), axis=1))
	z3 = np.matrix(a2.dot(theta2))
	a3 = np.matrix(sigmoid(z3))
	for n in range(m):
		yn = np.matrix(y[n,:])
		a3n = np.matrix(a3[n,:])
		J = J + ( -yn.T*np.log(a3n) - (1-yn).T*np.log(1-a3n) )
	J = (1/m) * J
	return J
#backpropagation/gradient descent
for j in range(epochs):
	for x in range(m): #for each training example
		#forward propagation
		a1 = np.matrix(np.concatenate((X[x,:], np.ones((1,1))), axis=1))
		z2 = np.matrix(a1.dot(theta1)) #1x3 * 3x3 = 1x3
		a2 = np.matrix(np.concatenate((sigmoid(z2), np.ones((1,1))), axis=1))
		z3 = np.matrix(a2.dot(theta2))
		a3 = np.matrix(sigmoid(z3))
		#backpropagation
		delta3 = np.matrix(a3 - y[x]) #1x1
		delta2 = np.matrix(np.multiply(theta2.dot(delta3), np.multiply(a2,(1-a2)).T)) #1x4
		theta1_grad += np.matrix((delta2[0:numHid, :].dot(a1))).T
		theta2_grad += np.matrix((delta3.dot(a2))).T #1x1 * 1x4 = 1x4

	theta1 += -1 * (1/m)*np.multiply(alpha, theta1_grad)
	theta2 += -1 * (1/m)*np.multiply(alpha, theta2_grad)
	#reset gradients
	theta1_grad = np.matrix(np.zeros((numIn+1, numHid)))
	theta2_grad = np.matrix(np.zeros((numHid + 1, numOut)))
	print(cost(X, y, theta1, theta2))

print("Results:\n")
a1 = np.matrix(np.concatenate((X, np.ones((4,1))), axis=1))
z2 = np.matrix(a1.dot(theta1))
a2 = np.matrix(np.concatenate((sigmoid(z2), np.ones((4,1))), axis=1))
z3 = np.matrix(a2.dot(theta2))
a3 = np.matrix(sigmoid(z3))
print(a3)