import numpy as np
from numpy.random import seed


class AdalineGD(object):
    """The ADAptive LInear NEuron classifier iterates on the perceptron formula by
    introducing a continuous linear activation function, rather than a unit step function.
    This means the cost function (here the sum of the squared errors) is both differentiable
    and convex, allowing us to utilize the powerful gradient descent algorithm to iterate on
    our weight values.

    Parameters
    ----------
    eta : float
        Learning rate [0,1].
    n_iter : int
        Number of iterations to cover dataset.

    Attributes
    ----------
    weights : array
        Weights after fitting.
    errors : list
        Number of misclassifications in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            Training vectors, with the shape defined by the
            number of samples and features
        y : array, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.weights = np.zeros(1 + X.shape[1])
        self.cost = []

        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights[1:]) +self.weights[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

class AdalineSGD(object):
    """The ADAptive LInear NEuron classifier iterates on the perceptron formula by
    introducing a continuous linear activation function, rather than a unit step function.
    This means the cost function (here the sum of the squared errors) is both differentiable
    and convex, allowing us to utilize the powerful gradient descent algorithm to iterate on
    our weight values. This version uses stochastic gradient descent.

    Parameters
    ----------
    eta : float
        Learning rate [0,1].
    n_iter : int
        Number of iterations to cover dataset.

    Attributes
    ----------
    weights : array
        Weights after fitting.
    errors : list
        Number of misclassifications in each epoch.
    shuffle : bool, optional
        Shuffles training data every epoch if True, to prevent cycles. Default is True.
    random_state : int
        Set random state for shuffling and initializing the weights.

    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.weights_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            Training vectors, with the shape defined by the
            number of samples and features
        y : array, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self._initialize_weights(X.shape[1])
        self.cost = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost.append(avg_cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights[1:]) +self.weights[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.weights_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, shape):
        """Initialize weights to zero"""
        self.weights = np.zeros(1 + shape)
        self.weights_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update weights"""
        output = self.net_input(xi)
        error = target - output
        self.weights[1:] += self.eta * xi.dot(error)
        self.weights[0] += self.eta * error
        cost = 0.5 * error ** 2.
        return cost