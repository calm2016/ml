from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
if __name__ == "__main__":
    house = datasets.load_boston()
    print house.feature_names
    print house.DESCR
    print house.data.shape
    data = house.data
    target = house.target
    plt.figure(figsize=(10, 10), facecolor='w')

    for f in range(data.shape[1]-1):
        plt.subplot(3, 4, f+1)
        plt.plot(data[:, f], target, 'b.')
        plt.title(house.feature_names[f], fontsize=12)
        # plt.xlabel('X', fontsize=16)
        # plt.ylabel('$', fontsize=16)

    plt.grid(True)
    plt.show()



