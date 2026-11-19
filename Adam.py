import numpy as np
class Adam:
    def __init__(self, params, beta1=0.5, beta2=0.999, eps=1e-8):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self, params, grads, lr):
        self.t += 1
        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.beta1 ** self.t)
            vh = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= lr * mh / (np.sqrt(vh) + self.eps)