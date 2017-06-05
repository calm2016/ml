print(__doc__)

# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
#
# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

quadratic_featurizer = PolynomialFeatures(degree=2)

# Use only one feature
X = np.arange(100).reshape(-1, 1)
data_Y = 3 + 2 * X + X * X + np.random.randint(1, 300, 100).reshape(-1, 1)
data_X = quadratic_featurizer.fit_transform(X)
print X
print data_Y
plt.scatter(X, data_Y, color='blue')

# model = Pipeline([('poly', PolynomialFeatures(degree=2)),
#                   ('linear', LinearRegression(fit_intercept=False))])

# Split the data into training/testing sets
data_X_train = data_X[:-20]
data_X_test = data_X[-20:]
# print data_X_train, data_X_test
#
# Split the targets into training/testing sets
data_y_train = data_Y[:-20]
data_y_test = data_Y[-20:]
# print data_y_train, data_y_test
# #

#
# # # # Create linear regression object
regr = LinearRegression()
#
# # Train the model using the training sets
regr.fit(data_X_train, data_y_train)
#
# # The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(data_X_test) - data_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(data_X_test, data_y_test))

# Plot outputs
print len(X[-20:])
print len(regr.predict(data_X_test))
plt.scatter(X[-20:], data_y_test, color='black')
plt.plot(X[-20:], regr.predict(data_X_test), color='blue', linewidth=3)

print (regr.predict(data_X_test) - data_y_test)/data_y_test
#
# # plt.xticks(())
# # plt.yticks(())
plt.show()
