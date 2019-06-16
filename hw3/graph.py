import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

data1 = np.array([
	[0,1],
	[2,2],
	[2,0],
])


data2 = np.array([
	[1,1],
	[3,1],
])


X,Y = data1.T
A,B = data2.T

plt.scatter(X,Y)
plt.scatter(A,B)
plt.show()