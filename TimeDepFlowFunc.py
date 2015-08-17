import random
from math import log, tanh
import matplotlib.pyplot as plt
import numpy as np
import time
import pylab

import pylab
import numpy

from scipy.integrate import odeint


plt.axis([0, 5, 0, 1])
plt.ion()
plt.show()

i = 5
while True:
	try:
		y = np.tanh(i)
		plt.plot(i, y)
		plt.draw()
		time.sleep(0.05)
		i -= 0.1
	except KeyboardInterrupt:
		break