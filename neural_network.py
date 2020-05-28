import numpy as np
import matplotlib.pylab as plt
def step_function(x):
    return np.array(x > 0, dtype = np.int)

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

#print(sigmoid(np.array([-1.0,1.0,2.0])))

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x,y)
y = sigmoid(x)
plt.plot(x,y)
y = ReLU(x)

plt.ylim(-0.1, 3.1)
plt.plot(x,y)
plt.show()
