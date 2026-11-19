import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def save_grid(G, path, z_fixed):
    imgs = (G.forward(z_fixed).reshape(-1, 28, 28) + 1) / 2
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for ax, img in zip(axes.ravel(), imgs):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def train(G, D, X, batch=64, epochs=100, eta_d=2e-4, eta_g=2e-4, eps=1e-7):
    z_fixed = np.random.randn(16, 100)

    for epoch in range(epochs):
        perm = np.random.permutation(len(X))
        d_sum = 0.0
        g_sum = 0.0
        steps = 0

        bar = tqdm(range(0, len(X) - batch + 1, batch),
                   desc=f"Epoch {epoch + 1}/{epochs}", ncols=110)

        for i in bar:
            real = X[perm[i:i + batch]]
            z = np.random.randn(batch, 100)
            fake = G.forward(z)

            pr = np.clip(D.forward(real), eps, 1 - eps)
            D.backward(-1.0 / pr / batch, eta_d)
            loss_real = -np.mean(np.log(pr))

            pf = np.clip(D.forward(fake), eps, 1 - eps)
            D.backward(1.0 / (1 - pf) / batch, eta_d)
            loss_fake = -np.mean(np.log(1 - pf))

            pg = np.clip(D.forward(fake), eps, 1 - eps)
            dfake = D.backward(-1.0 / pg / batch, eta_d, update=False)
            G.backward(dfake, eta_g)
            loss_g = -np.mean(np.log(pg))

            d_sum += loss_real + loss_fake
            g_sum += loss_g
            steps += 1

            bar.set_postfix(D=f"{d_sum / steps:.3f}", G=f"{g_sum / steps:.3f}")

        if epoch % 5 == 0:
            save_grid(G, f"epoch_{epoch}.png", z_fixed)
            np.savez("gan_weights.npz",
                     **{f"G_{k}": v for k, v in G.p.items()},
                     **{f"D_{k}": v for k, v in D.p.items()})