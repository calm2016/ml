from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
if __name__ == "__main__":
    house = datasets.load_boston()
    # print house.feature_names
    # print house.DESCR
    # print house.data.shape
    data = house.data
    target = house.target
    plt.figure(figsize=(10, 10), facecolor='w')

    for f in range(data.shape[1]-1):
        plt.subplot(3, 4, f+1)
        x = data[:, f].reshape([-1, 1])
        plt.plot(x, target, 'b.')
        plt.title(house.feature_names[f], fontsize=12)
        x_train, x_test, y_train, y_test = train_test_split(x, target, random_state=1, train_size=0.8)
        lr = linear_model.LinearRegression()
        lr.fit(x_train, y_train)
        print('Coefficients: \n', lr.coef_)
        print("Mean squared error: %.2f" % np.mean((lr.predict(x_test) - y_test) ** 2))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % lr.score(x_test, y_test))
        plt.plot(x_test, lr.predict(x_test), color='red', linewidth=3)

    plt.grid(True)
    plt.show()



