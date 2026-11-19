import numpy as np
from Adam import Adam
from Activations import LayerNorm,LeakyReLU,Sigmoid
class Discriminator:
    def __init__(self):
        sizes = [784, 1024, 512, 256, 128, 1]
        self.p = {}
        for i in range(5):
            scale = np.sqrt(2 / sizes[i]) if i < 4 else np.sqrt(1 / sizes[i])
            self.p[f"w{i+1}"] = np.random.randn(sizes[i], sizes[i + 1]) * scale
            self.p[f"b{i+1}"] = np.zeros((1, sizes[i + 1]))

        self.f1 = LeakyReLU()
        self.N2 = LayerNorm()
        self.f2 = LeakyReLU()
        self.f3 = LeakyReLU()
        self.N4 = LayerNorm()
        self.f4 = LeakyReLU()
        self.f5 = Sigmoid()
        self.opt = Adam(self.p)

    def forward(self, x):
        p = self.p
        self.x = np.asarray(x).reshape(-1, 784)

        self.a1 = self.x @ p["w1"] + p["b1"]
        self.y1 = self.f1.forward(self.a1)

        self.a2 = self.y1 @ p["w2"] + p["b2"]
        self.an2, self.cache2 = self.N2.forward(self.a2)
        self.y2 = self.f2.forward(self.an2)

        self.a3 = self.y2 @ p["w3"] + p["b3"]
        self.y3 = self.f3.forward(self.a3)

        self.a4 = self.y3 @ p["w4"] + p["b4"]
        self.an4, self.cache4 = self.N4.forward(self.a4)
        self.y4 = self.f4.forward(self.an4)

        self.a5 = self.y4 @ p["w5"] + p["b5"]
        self.y5 = self.f5.forward(self.a5)

        return self.y5

    def backward(self, grad_out, eta, update=True):
        p = self.p
        g = {}

        da5 = self.f5.backward(self.y5, np.asarray(grad_out).reshape(-1, 1))
        g["w5"] = self.y4.T @ da5
        g["b5"] = da5.sum(axis=0, keepdims=True)

        dy4 = da5 @ p["w5"].T
        dan4 = self.f4.backward(self.an4, dy4)
        da4 = self.N4.backward(dan4, self.cache4)
        g["w4"] = self.y3.T @ da4
        g["b4"] = da4.sum(axis=0, keepdims=True)

        dy3 = da4 @ p["w4"].T
        da3 = self.f3.backward(self.a3, dy3)
        g["w3"] = self.y2.T @ da3
        g["b3"] = da3.sum(axis=0, keepdims=True)

        dy2 = da3 @ p["w3"].T
        dan2 = self.f2.backward(self.an2, dy2)
        da2 = self.N2.backward(dan2, self.cache2)
        g["w2"] = self.y1.T @ da2
        g["b2"] = da2.sum(axis=0, keepdims=True)

        dy1 = da2 @ p["w2"].T
        da1 = self.f1.backward(self.a1, dy1)
        g["w1"] = self.x.T @ da1
        g["b1"] = da1.sum(axis=0, keepdims=True)

        dx = da1 @ p["w1"].T

        if update:
            self.opt.step(self.p, g, eta)

        return dx