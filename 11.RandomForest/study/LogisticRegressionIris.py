from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
x1 = iris.data[:, 0]
x2 = iris.data[:, 1]
y = np.array(iris.target)
# print x, y
# stack = np.stack((x1, x2, y), -1)
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
plt.scatter(x1, x2, c=y, cmap=cm_light, marker='.')
x = np.stack((x1, x2), -1)
x_train, x_test, y_train, y_test = train_test_split(iris.data, y, train_size=0.7, random_state=1)

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_hat = lr.predict(x_test)
# print np.stack((y_test, y_hat), -1)
result = (100 * np.mean(y_hat == y_test.ravel()))
print "LogisticRegression accuracy:", result
# plt.show()

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
y_hat = tree.predict(x_test)
result = (100 * np.mean(y_hat == y_test.ravel()))
print "DecisionTreeClassifier accuracy:", result

rfc = RandomForestClassifier(n_estimators=25)
rfc.fit(x_train, y_train)
y_hat = rfc.predict(x_test)
result = (100 * np.mean(y_hat == y_test.ravel()))
print "RandomForestClassifier accuracy:", result

C = 1.0  # SVM regularization parameter
clf = svm.SVC(kernel='linear', C=C)
clf.fit(x_train, y_train)
y_hat = clf.predict(x_test)
# print np.stack((y_hat, y_test), -1)
result = (100 * np.mean(y_hat == y_test.ravel()))
print "SVM accuracy:", result
