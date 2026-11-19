import numpy as np

from Discriminator import Discriminator
from Generator import Generator
from trainer import train

DATA_DIR = r"C:\Users\n\Downloads\archive"

X = np.fromfile(DATA_DIR + r"\train-images-idx3-ubyte", dtype=np.uint8, offset=16)
X = X.reshape(-1, 784).astype(np.float32) / 127.5 - 1

G = Generator()
D = Discriminator()

train(G, D, X, batch=64, epochs=100, eta_d=2e-4, eta_g=2e-4, eps=1e-7)