import numpy as np
def sigmoid(x):
	return np.matrix(1.0 / (1.0 + np.exp(-x)))