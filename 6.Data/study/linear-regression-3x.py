print(__doc__)

# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
#
# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
quadratic_featurizer = PolynomialFeatures(degree=2)


# Use only one feature
data_X = np.arange(0, 10, 0.1).reshape(-1, 1)
data_Y = 3*data_X
# data_X = quadratic_featurizer.fit_transform(data_X)
# print data_X
# print data_Y

# Split the data into training/testing sets
data_X_train = data_X[:-20]
data_X_test = data_X[-20:]
#
# Split the targets into training/testing sets
data_y_train = data_Y[:-20]
data_y_test = data_Y[-20:]
print data_y_train, data_y_test
#
plt.plot(data_X, data_Y)

# # # Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(data_X_train, data_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(data_X_test) - data_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(data_X_test, data_y_test))

# Plot outputs
plt.scatter(data_X_test, data_y_test,  color='black')
plt.plot(data_X_test, regr.predict(data_X_test), color='blue',
         linewidth=3)

# plt.xticks(())
# plt.yticks(())
plt.show()
