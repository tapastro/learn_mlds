import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from perceptron import Perceptron


def plot_decision_regions(X, y, classifier, resolution=0.01):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)

X = df.iloc[0:100, [0, 2]].values
y = np.where(df.iloc[0:100, 4].values == 'Iris-setosa', -1, 1)

fig, ax = plt.subplots(1, 1)

ax.scatter(X[:50, 0], X[:50, 1], color='green', marker='o', label='Setosa')
ax.scatter(X[50:100, 0], X[50:100, 1], color='orange', marker='x', label='Versicolor')


ax.set_xlabel('Sepal Length')
ax.set_ylabel('Petal Length')
ax.set_title('Data Space')

plt.legend(loc='best')
plt.show()
plt.clf()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors) + 1), ppn.errors)
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
plt.clf()


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='best')
plt.show()