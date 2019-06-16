import numpy as np
import pandas as pd
import size

from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet


#part a)

X_train = pd.read_csv('airline_delays_Xtrain.csv')
y_train = pd.read_csv('airline_delays_ytrain.csv')

ridge_lambdas = np.logspace(start=4, stop=14, num=22, base=10)

ridge_exps = np.log10(ridge_lambdas)

ridge_coefs = np.empty((ridge_lambdas.shape[0], X_train.shape[1]))


for i in range(0,length(ridge_lambdas)-1):
	clf = Ridge(alpha=ridge_lambdas[i])
	clf.fit(X_train, y_train)


plt.plot(ridge_exps,ridge_coefs)

plt.xlabel('Log lambda')
plt.ylabel('Coefficient Vlue')

plt.title('Log Lambda vs Coefficient')

plt.show()