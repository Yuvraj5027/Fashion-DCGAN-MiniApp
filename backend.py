import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

LATENT_DIM = 100
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)


# ---------- Dataset loader ----------
def synthetic_fashion(n):
    # Fallback if Fashion-MNIST cannot be downloaded: random shirt / trouser / shoe-like shapes
    X = torch.full((n, 1, 28, 28), -1.0)
    for i in range(n):
        t = np.random.randint(3)
        v = np.random.uniform(0.3, 1.0) * 2 - 1
        if t == 0:
            X[i, 0, 6:24, 7:21] = v; X[i, 0, 6:12, 3:25] = v
        elif t == 1:
            X[i, 0, 4:26, 8:13] = v; X[i, 0, 4:26, 15:20] = v
        else:
            X[i, 0, 16:23, 3:25] = v; X[i, 0, 11:16, 3:10] = v
    return TensorDataset(X, torch.zeros(n))


def load_data(subset_size=3000, batch_size=128):
    try:
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        full = datasets.FashionMNIST("data", train=True, download=True, transform=tfm)
        ds = Subset(full, torch.randperm(len(full))[:subset_size].tolist())
    except Exception as e:
        print("Fashion-MNIST download failed, using synthetic data:", e)
        ds = synthetic_fashion(subset_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)


# ---------- Models ----------
class Generator(nn.Module):
    def __init__(self, z=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z, 64, 7, 1, 0, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, x):
        return self.net(x.view(x.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 1, 7, 1, 0, bias=False))

    def forward(self, x):
        return self.net(x).view(-1)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02); nn.init.zeros_(m.bias)


# ---------- Training ----------
criterion = nn.BCEWithLogitsLoss()
fixed_noise = torch.randn(32, LATENT_DIM)


def train_step(G, D, optG, optD, real):
    b = real.size(0)
    real_lbl, fake_lbl = torch.ones(b), torch.zeros(b)
    fake = G(torch.randn(b, LATENT_DIM))
    optD.zero_grad()
    d_loss = criterion(D(real), real_lbl) + criterion(D(fake.detach()), fake_lbl)
    d_loss.backward(); optD.step()
    optG.zero_grad()
    g_loss = criterion(D(fake), real_lbl)
    g_loss.backward(); optG.step()
    return d_loss.item(), g_loss.item()


def train(epochs=8, subset_size=3000, batch_size=128, lr=2e-4, on_epoch=None):
    loader = load_data(subset_size, batch_size)
    G, D = Generator(), Discriminator()
    G.apply(init_weights); D.apply(init_weights)
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    history = {"d_loss": [], "g_loss": [], "samples": []}
    for ep in range(1, epochs + 1):
        dl, gl = [], []
        for real, _ in loader:
            d, g = train_step(G, D, optG, optD, real)
            dl.append(d); gl.append(g)
        history["d_loss"].append(float(np.mean(dl)))
        history["g_loss"].append(float(np.mean(gl)))
        G.eval()
        with torch.no_grad():
            history["samples"].append(G(fixed_noise).cpu())
        G.train()
        torch.save({"epoch": ep, "G": G.state_dict(), "D": D.state_dict(),
                    "d_loss": history["d_loss"], "g_loss": history["g_loss"]},
                   f"{CKPT_DIR}/dcgan_epoch_{ep}.pt")
        if on_epoch:
            on_epoch(ep, epochs, history)
    return G, D, history


def generate(G, n=16):
    G.eval()
    with torch.no_grad():
        return G(torch.randn(n, LATENT_DIM)).cpu()
