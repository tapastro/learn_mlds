import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from adaline import AdalineGD, AdalineSGD


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

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost) + 1), np.log10(ada1.cost), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-Squared Error)')
ax[0].set_title(f'Adaline - Learning Rate {ada1.eta}')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost) + 1), np.log10(ada2.cost), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-Squared Error)')
ax[1].set_title(f'Adaline - Learning Rate {ada2.eta}')
plt.show()
plt.clf()

"""Learning rate 0.01 overshoots the global minimum, while 0.0001 will take many iterations
to converge. If we normalize our input into a standard normal, we can gain some intuition on
learning rates for classes of problems.
"""
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

X_std = np.copy(X)
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

ada3 = AdalineGD(n_iter=15, eta=0.01).fit(X_std, y)
ax[0].plot(range(1, len(ada3.cost) + 1), np.log10(ada3.cost), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-Squared Error)')
ax[0].set_title(f'Adaline - Learning Rate {ada3.eta}')
plot_decision_regions(X_std, y, classifier=ada3)
ax[1].set_title('Adaline - Gradient Descent')
ax[1].set_xlabel('Sepal Length [standardized]')
ax[1].set_ylabel('Petal Length [standardized]')
ax[1].legend(loc='best')
plt.show()
plt.clf()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

X_std = np.copy(X)
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

ada4 = AdalineSGD(n_iter=15, eta=0.01, random_state=1).fit(X_std, y)
ax[0].plot(range(1, len(ada4.cost) + 1), ada4.cost, marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Average Cost')
ax[0].set_title(f'AdalineSGD - Learning Rate {ada4.eta}')
plot_decision_regions(X_std, y, classifier=ada4)
ax[1].set_title('Adaline - Stochastic Gradient Descent')
ax[1].set_xlabel('Sepal Length [standardized]')
ax[1].set_ylabel('Petal Length [standardized]')
ax[1].legend(loc='best')
plt.show()



