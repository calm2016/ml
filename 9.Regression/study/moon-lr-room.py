from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

import numpy as np

if __name__ == "__main__":
    house = datasets.load_boston()
    index = 5  # RM, average number of rooms per dwelling
    data = house.data
    target = house.target

    x = data[:, index].reshape([-1, 1])

    plt.plot(x, target, 'b.')
    plt.title(house.feature_names[index], fontsize=12)
    x_train, x_test, y_train, y_test = train_test_split(x, target, random_state=1, train_size=0.8)

    order = x_test.argsort(axis=0)
    y_test = y_test[order]
    x_test = x_test[order].reshape(-1, 1)

    models = [
        Pipeline([
            ('linear', LinearRegression(fit_intercept=False))]),
        Pipeline([
            ('poly', PolynomialFeatures()),
            ('linear', LinearRegression(fit_intercept=False))])
    ]
    colors = ['blue', 'red']

    for i, m in enumerate(models):
        m.fit(x_train, y_train)
        # print('Coefficients: \n', m.coef_)
        print x_test.shape
        predicts = m.predict(x_test)
        print("Mean squared error: %.2f" % np.mean((predicts - y_test) ** 2))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % m.score(x, target))
        plt.plot(x_test, m.predict(x_test), color=colors[i], linewidth=1)

    plt.grid(True)
    plt.show()



