import numpy as np
class LeakyReLU:
    def __init__(self, slope=0.2):
        self.slope = slope

    def forward(self, x):
        return np.where(x > 0, x, self.slope * x)

    def backward(self, x, grad):
        return grad * np.where(x > 0, 1.0, self.slope)


class Sigmoid:
    def forward(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def backward(self, y, grad):
        return grad * y * (1 - y)


class Tanh:
    def forward(self, x):
        return np.tanh(x)

    def backward(self, y, grad):
        return grad * (1 - y ** 2)


class LayerNorm:
    def forward(self, x):
        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        sigma = np.sqrt(var + 1e-8)
        x_norm = (x - mean) / sigma
        return x_norm, (x_norm, sigma)

    def backward(self, grad, cache):
        x_norm, sigma = cache
        n = grad.shape[1]
        return (1 / (n * sigma)) * (
            n * grad
            - np.sum(grad, axis=1, keepdims=True)
            - x_norm * np.sum(grad * x_norm, axis=1, keepdims=True)
        )