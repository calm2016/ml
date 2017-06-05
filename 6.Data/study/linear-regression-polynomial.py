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
diabetes_X = np.arange(0, 10, 0.1).reshape(-1, 1)
diabetes_Y = 3*diabetes_X
# diabetes_X = quadratic_featurizer.fit_transform(diabetes_X)
# print diabetes_X
# print diabetes_Y

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
#
# Split the targets into training/testing sets
diabetes_y_train = diabetes_Y[:-20]
diabetes_y_test = diabetes_Y[-20:]
print diabetes_y_train, diabetes_y_test
#
plt.plot(diabetes_X, diabetes_Y)

# # # Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

# plt.xticks(())
# plt.yticks(())
plt.show()
