import numpy as np


class Perceptron(object):
    """The perceptron classifier receives an input vector X and applies a weight vector w to it.
    The net input Sigma(wX) is fed to the activation function, which for a perceptron is the unit
    step function, outputting -1 or 1 for the predicted class label. The perceptron is only
    guaranteed to converge if the classes are linearly separable *and* the learning rate is
    sufficiently small to sample the boundary space separating the classes in X space.

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
    def __init__(self, eta=0.01, n_iter=10):
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
        self.errors = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights[1:]) +self.weights[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)