"""W03. Rosenblatt perceptron.

Interface (Appendix A):
    perceptron_update(w, b, x, y, lr) -> (w, b)
        One update on one example; y is -1 or +1. If y * (w @ x + b) <= 0 then
        w <- w + lr * y * x and b <- b + lr * y, otherwise w, b are returned unchanged.
    class Perceptron(lr=1.0, max_epochs=100, seed=0)
        fit(X, y) with y in {-1, +1}; returns self. Attributes w_, b_.
        predict(X) -> array of -1 / +1.
"""
import numpy as np


def perceptron_update(w, b, x, y, lr):
    if y * (w @ x + b) <= 0:
        w = w + lr * y * x
        b = b + lr * y
    return w, b


class Perceptron:
    def __init__(self, lr=1.0, max_epochs=100, seed=0):
        self.lr, self.max_epochs, self.seed = lr, max_epochs, seed

    def fit(self, X, y):
        # Initualize
        X = np.asarray(X, dtype = float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        
        self.w_ = np.zeros(X.shape[1])
        self.b_ = 0.0
        
        # Loop every epoch and count errors
        for _ in range(self.max_epochs):
            errors = 0
            for i in rng.permutation(len(X)):
                if y[i] * (self.w_ @ X[i] + self.b_) <= 0:
                    errors += 1
                self.w_, self.b_ = perceptron_update(self.w_, self.b_, X[i], y[i], self.lr)
            if errors == 0:
                break
        return self
        

    def predict(self, X):
        scores = np.asarray(X, dtype = float) @ self.w_ + self.b_
        return np.where(scores >= 0, 1, -1)
