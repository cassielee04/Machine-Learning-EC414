#Seung Hee Lee Homework 2 Problem 3
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import statsmodels.api as sm
import statsmodels.formula.api as smf

#part a)
greenland_df = pd.read_csv('greenland_mass.csv', index_col=0)
example = greenland_df.head()

x = greenland_df['time'].values
y = greenland_df['mass_diff'].values
b = sm.add_constant(greenland_df['time'].values)



plt.plot(x,y)
plt.xlabel('year')
plt.ylabel('Mass Difference in Gigatonic')

plt.title('Greenland Mass')

plt.show()


#part b)

X = greenland_df[['time']]
Y = greenland_df[['mass_diff']]

nrow = X.shape[0]

onesX = np.ones((nrow,1))

AugX = np.concatenate((onesX,X),axis = 1)

W = np.dot(np.linalg.inv(np.dot(AugX.T,AugX)),np.dot(AugX.T,Y))
print(W)

pred2018 = W[1]*2018 + W[0]

print("2018 value:",pred2018)


#part c)

prediction = x*W[1] + W[0]

plt.plot(x,y, label='Data')
plt.plot(x,prediction, label='OLS fit')

plt.xlabel('year')
plt.ylabel('Mass Difference in Gigatonic')

plt.title('Greenland Mass')

plt.legend()
plt.show()










