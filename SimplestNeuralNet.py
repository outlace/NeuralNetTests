import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#Our training data
X = np.matrix('0 1;1 1')
y = np.matrix('1;0')
#Let's "Randomly" initialize weight to 5, just so we can see gradient descent at work
#weights = 6 * np.matrix(np.random.randn(2,1))
ep_init = 1.73
weights = np.matrix(np.random.normal(0,5, (2,1)))
	#( 3 * np.sqrt ( 6 / ( 1 ) ) * np.random.randn( 2, 1 ) )
#sigmoid function
def sigmoid(x):
	return np.matrix(1.0 / (1.0 + np.exp(-x)))

#run the neural net forward
def run(X, weights):
	return sigmoid(X * weights) #1x2 * 2x2 = 1x1 matrix

#Our cost function
def cost(X, y, weights):
	nn_output = run(X, weights)
	m = X.shape[0] #num training examples, 2
	#return np.sum((1/m) * np.square(nn_output - y))
	return np.sum( -y.T*np.log(nn_output) - (1-y).T*np.log(1-nn_output));
'''
print('Initial Weight: %s\n' % weights)
print('Cost Before Gradient Descent: %s \n' % cost(X, y, weights))

#Gradient Descent
alpha = 0.05 #learning rate
epochs = 12000 #num iterations
for i in range(epochs):
	cost_derivative1 = np.sum(1 * np.multiply((run(X, weights) - y), np.multiply(run(X, weights), (1 - run(X, weights)))))
	cost_derivative2 = np.sum(X[:,0] * np.multiply((run(X, weights) - y), np.multiply(run(X, weights), (1 - run(X, weights)))).T)
	weights[0] = weights[0] - alpha * cost_derivative1
	weights[1] = weights[1] - alpha * cost_derivative2
print('Final Weight: %s\n' % weights)
print('Final Cost: %s \n' % cost(X, y, weights))
print('Result:\n')
print(np.round(run(X, weights)))
print('Expected Result\n')
print(y)
'''
#theta1 = np.random.normal(0, 5, 100)
#theta2 = np.random.normal(0, 5, 100)
theta1 = np.random.permutation(np.linspace(-15, 15, 500))
theta2 = np.random.permutation(np.linspace(-15, 15, 500))
#Axes3D.plot(weights1, weights2, )
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
zs = [cost(X, y, np.matrix(theta).T) for theta in zip(theta1, theta2)]
#theta1, theta = np.meshgrid(theta1, theta2)
ax.plot_trisurf(theta1, theta2, zs, linewidth=.02, cmap=cm.jet)

ax.set_xlabel('Theta1')
ax.set_ylabel('Theta2')
ax.set_zlabel('Cost')

plt.show()