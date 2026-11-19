"""
Inference-only post-training evaluation for the supplied NumPy GAN.
This script loads the trained .npz weights and performs forward passes only.
It does NOT construct an optimizer, compute gradients, update parameters, or train.

Dependencies:
    pip install numpy pandas matplotlib scipy

Examples:
    python evaluate_gan.py --checkpoint gan_weights_99.npz --fashion-csv fashion-mnist_train.csv
    python evaluate_gan.py --checkpoint gan_weights_99.npz --mnist-idx train-images-idx3-ubyte

The original training code reads an IDX file and scales pixels as uint8/127.5 - 1.
Use the same dataset domain used for training. The supplied CSV is Fashion-MNIST.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

class GANInference:
    def __init__(self, checkpoint):
        ck = np.load(checkpoint, allow_pickle=False)
        self.G, self.D = {}, {}
        specs = {
            "G": {"w1":(100,256),"b1":(1,256),"w2":(256,1024),"b2":(1,1024),
                  "w3":(1024,512),"b3":(1,512),"w4":(512,784),"b4":(1,784)},
            "D": {"w1":(784,1024),"b1":(1,1024),"w2":(1024,512),"b2":(1,512),
                  "w3":(512,256),"b3":(1,256),"w4":(256,128),"b4":(1,128),
                  "w5":(128,1),"b5":(1,1)}
        }
        expected = sum(len(v) for v in specs.values())
        for prefix, shapes in specs.items():
            target = self.G if prefix == "G" else self.D
            for key, shape in shapes.items():
                name = f"{prefix}_{key}"
                if name not in ck.files: raise KeyError(f"Checkpoint missing {name}")
                if ck[name].shape != shape:
                    raise ValueError(f"{name}: expected {shape}, got {ck[name].shape}")
                if not np.isfinite(ck[name]).all(): raise ValueError(f"{name} contains NaN/Inf")
                target[key] = ck[name].astype(np.float64, copy=True)
        if len(ck.files) != expected:
            raise ValueError(f"Expected {expected} arrays, found {len(ck.files)}")
        print(f"Checkpoint validated: {expected}/{expected} arrays, all shapes and values valid.")

    @staticmethod
    def lrelu(x): return np.where(x > 0, x, 0.2*x)
    @staticmethod
    def layer_norm(x):
        return (x-x.mean(axis=1,keepdims=True))/np.sqrt(x.var(axis=1,keepdims=True)+1e-8)

    def generate(self, z):
        p=self.G; z=np.asarray(z,dtype=np.float64).reshape(-1,100)
        h=self.lrelu(z@p["w1"]+p["b1"])
        h=self.lrelu(self.layer_norm(h@p["w2"]+p["b2"]))
        h=self.lrelu(h@p["w3"]+p["b3"])
        return np.tanh(h@p["w4"]+p["b4"])

    def discriminate(self, x):
        p=self.D; x=np.asarray(x,dtype=np.float64).reshape(-1,784)
        h=self.lrelu(x@p["w1"]+p["b1"])
        h=self.lrelu(self.layer_norm(h@p["w2"]+p["b2"]))
        h=self.lrelu(h@p["w3"]+p["b3"])
        h=self.lrelu(self.layer_norm(h@p["w4"]+p["b4"]))
        logits=np.clip(h@p["w5"]+p["b5"],-500,500)
        return 1/(1+np.exp(-logits))

def image_grid(images, path, title, cols=8):
    rows=int(np.ceil(len(images)/cols))
    fig,axs=plt.subplots(rows,cols,figsize=(cols*1.35,rows*1.45))
    axs=np.asarray(axs).reshape(-1)
    for ax,img in zip(axs,images):
        ax.imshow(np.clip((img.reshape(28,28)+1)/2,0,1),cmap="gray",vmin=0,vmax=1)
        ax.axis("off")
    for ax in axs[len(images):]: ax.axis("off")
    fig.suptitle(title,fontsize=15,fontweight="bold")
    fig.tight_layout()
    fig.savefig(path,dpi=300,bbox_inches="tight")
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    src=ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fashion-csv")
    src.add_argument("--mnist-idx")
    ap.add_argument("--out",default="gan_post_training_evaluation")
    ap.add_argument("--seed",type=int,default=20261119)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    gan=GANInference(args.checkpoint)

    if args.fashion_csv:
        df=pd.read_csv(args.fashion_csv)
        if df.shape[1]!=785: raise ValueError("Fashion-MNIST CSV must have label + 784 pixel columns")
        X=df.iloc[:,1:].to_numpy(dtype=np.float64)/127.5-1
        domain="Fashion-MNIST CSV"
        real_labels=df.iloc[:,0].to_numpy()
    else:
        raw=np.fromfile(args.mnist_idx,dtype=np.uint8,offset=16)
        if raw.size%784: raise ValueError("IDX image payload is not divisible into 784-pixel images")
        X=raw.reshape(-1,784).astype(np.float64)/127.5-1
        domain="MNIST IDX"
        real_labels=None

    n=min(1000,len(X))
    real=X[rng.choice(len(X),size=n,replace=False)]
    fake=gan.generate(rng.standard_normal((n,100)))
    assert np.isfinite(fake).all() and fake.shape==(n,784)

    # A. 64 generated samples.
    image_grid(gan.generate(rng.standard_normal((64,100))),out/"generated_images.png",
               "Generated Samples from the Trained NumPy GAN (64)",8)

    # B. Discriminator score distributions.
    rs=gan.discriminate(real).ravel(); fs=gan.discriminate(fake).ravel()
    plt.figure(figsize=(9,5))
    bins=np.linspace(0,1,31)
    plt.hist(rs,bins=bins,density=True,alpha=.58,label=f"Real ({domain})")
    plt.hist(fs,bins=bins,density=True,alpha=.58,label="Generated")
    plt.axvline(.5,ls="--",lw=1.5,label="Threshold = 0.5")
    plt.xlabel("Discriminator output probability"); plt.ylabel("Density")
    plt.title("Real vs Generated Discriminator Scores"); plt.xlim(0,1)
    plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(out/"real_vs_fake_scores.png",dpi=300); plt.close()

    # C. Threshold metrics: real=1, fake=0.
    real_acc=float(np.mean(rs>=.5)); fake_acc=float(np.mean(fs<.5))
    balanced=(real_acc+fake_acc)/2
    plt.figure(figsize=(7.5,5))
    bars=plt.bar(["Real accuracy","Fake accuracy","Balanced accuracy"],[real_acc,fake_acc,balanced])
    plt.ylim(0,1.05); plt.ylabel("Accuracy")
    plt.title("Discriminator Accuracy (threshold = 0.5)")
    plt.bar_label(bars,fmt="%.3f",padding=4); plt.grid(axis="y",alpha=.25)
    plt.tight_layout(); plt.savefig(out/"discriminator_accuracy.png",dpi=300); plt.close()

    # D. Linear latent interpolation.
    z1,z2=rng.standard_normal(100),rng.standard_normal(100)
    ts=np.linspace(0,1,12)
    zi=np.array([(1-t)*z1+t*z2 for t in ts])
    ims=gan.generate(zi)
    fig,axs=plt.subplots(2,6,figsize=(12,4.8))
    for ax,im,t in zip(axs.ravel(),ims,ts):
        ax.imshow(np.clip((im.reshape(28,28)+1)/2,0,1),cmap="gray",vmin=0,vmax=1)
        ax.set_title(f"t={t:.2f}"); ax.axis("off")
    fig.suptitle("Latent-Space Linear Interpolation",fontweight="bold")
    fig.tight_layout(); fig.savefig(out/"latent_interpolation.png",dpi=300); plt.close(fig)

    # E. One-coordinate traversals; all other latent coordinates are fixed.
    base=rng.standard_normal(100); dims=[0,10,25,50,75,99]; steps=np.linspace(-3,3,9)
    fig,axs=plt.subplots(len(dims),len(steps),figsize=(15,10))
    for r,d in enumerate(dims):
        zz=np.repeat(base[None,:],len(steps),axis=0); zz[:,d]=steps
        for c,(ax,im) in enumerate(zip(axs[r],gan.generate(zz))):
            ax.imshow(np.clip((im.reshape(28,28)+1)/2,0,1),cmap="gray",vmin=0,vmax=1)
            ax.axis("off")
            if r==0: ax.set_title(f"{steps[c]:.1f}")
        axs[r,0].set_ylabel(f"z[{d}]",rotation=0,labelpad=24,va="center")
    fig.suptitle("Latent-Dimension Traversals",fontweight="bold")
    fig.tight_layout(); fig.savefig(out/"latent_variation.png",dpi=300); plt.close(fig)

    # F. Pixel intensity and per-image statistics.
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    bins=np.linspace(-1,1,61)
    axs[0].hist(real.ravel(),bins=bins,density=True,alpha=.58,label="Real")
    axs[0].hist(fake.ravel(),bins=bins,density=True,alpha=.58,label="Generated")
    axs[0].set(title="Pixel-Intensity Distribution",xlabel="Pixel intensity [-1,1]",ylabel="Density")
    axs[0].legend(frameon=False)
    axs[1].boxplot([real.mean(1),fake.mean(1),real.std(1),fake.std(1)],
                   tick_labels=["Real mean","Fake mean","Real SD","Fake SD"],showfliers=False)
    axs[1].set(title="Per-Image Pixel Statistics",ylabel="Value on [-1,1]")
    axs[1].tick_params(axis="x",rotation=15)
    fig.tight_layout(); fig.savefig(out/"distribution_analysis.png",dpi=300); plt.close(fig)

    # Real sample grid for direct qualitative comparison.
    ridx=rng.choice(n,size=min(64,n),replace=False)
    image_grid(real[ridx],out/"provided_real_samples.png",f"Real Samples: {domain}",8)

    # Descriptive diversity indicators; not a semantic mode-collapse test.
    fake_pw=pdist(fake[:min(500,n)])
    real_pw=pdist(real[:min(500,n)])
    result={
      "dataset_used":domain,"real_sample_count":n,"fake_sample_count":n,"seed":args.seed,
      "real_accuracy":real_acc,"fake_accuracy":fake_acc,"balanced_accuracy":balanced,
      "mean_real_discriminator_score":float(rs.mean()),"std_real_discriminator_score":float(rs.std()),
      "mean_fake_discriminator_score":float(fs.mean()),"std_fake_discriminator_score":float(fs.std()),
      "real_mean_across_sample_pixel_std":float(real.std(axis=0).mean()),
      "fake_mean_across_sample_pixel_std":float(fake.std(axis=0).mean()),
      "real_pairwise_l2_mean_first_500":float(real_pw.mean()),
      "fake_pairwise_l2_mean_first_500":float(fake_pw.mean()),
      "fake_fraction_near_constant_std_below_0_03":float(np.mean(fake.std(1)<.03)),
      "standard_FID":"Not computed.",
      "feature_based_FID":"Not computed: provide a domain-appropriate pretrained Fashion-MNIST/MNIST feature extractor.",
      "notes":"Pixel-space statistics are not FID. Use the same dataset domain used during training."
    }
    (out/"evaluation_metrics.json").write_text(json.dumps(result,indent=2))
    pd.DataFrame([result]).to_csv(out/"evaluation_metrics.csv",index=False)
    print(json.dumps(result,indent=2))
    print(f"Saved plots and metrics to: {out.resolve()}")

if __name__=="__main__": main()
