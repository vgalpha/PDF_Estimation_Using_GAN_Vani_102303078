import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader, TensorDataset

r = 102303078
a_r = 0.5 * (r % 7)       # = 1.5
b_r = 0.3 * (r % 5 + 1)   # = 1.2

print("a_r =", a_r)
print("b_r =", b_r)

df = pd.read_csv("data.csv", encoding="latin1", low_memory=False)
x = df["no2"].dropna().values.astype(np.float32)
x = x[x > 0]

z = x + a_r * np.sin(b_r * x)
print("z shape:", z.shape)
print("z min:", z.min(), "z max:", z.max())

z_mean = z.mean()
z_std = z.std()
z_norm = (z - z_mean) / z_std

z_tensor = torch.tensor(z_norm.reshape(-1, 1), dtype=torch.float32)
dataset = TensorDataset(z_tensor)
loader = DataLoader(dataset, batch_size=256, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, noise):
        return self.model(noise)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

G = Generator().to(device)
D = Discriminator().to(device)

loss_fn = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=0.0003)
opt_D = torch.optim.Adam(D.parameters(), lr=0.0003)

g_loss_list = []
d_loss_list = []

epochs = 3000

for epoch in range(epochs):
    g_total = 0.0
    d_total = 0.0
    n_batches = 0

    for batch in loader:
        real_data = batch[0].to(device)
        bs = real_data.size(0)

        real_labels = torch.ones(bs, 1, device=device)
        fake_labels = torch.zeros(bs, 1, device=device)

        # train discriminator
        noise = torch.randn(bs, 1, device=device)
        fake_data = G(noise).detach()

        loss_real = loss_fn(D(real_data), real_labels)
        loss_fake = loss_fn(D(fake_data), fake_labels)
        d_loss = loss_real + loss_fake

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # train generator
        noise = torch.randn(bs, 1, device=device)
        fake_data = G(noise)
        g_loss = loss_fn(D(fake_data), real_labels)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        g_total += g_loss.item()
        d_total += d_loss.item()
        n_batches += 1

    g_epoch_loss = g_total / n_batches
    d_epoch_loss = d_total / n_batches
    g_loss_list.append(g_epoch_loss)
    d_loss_list.append(d_epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}  D_loss: {d_epoch_loss:.4f}  G_loss: {g_epoch_loss:.4f}")


G.eval()
with torch.no_grad():
    noise = torch.randn(10000, 1, device=device)
    z_fake_norm = G(noise).cpu().numpy().flatten()

z_fake = z_fake_norm * z_std + z_mean

kde_real = gaussian_kde(z)
kde_fake = gaussian_kde(z_fake)

z_vals = np.linspace(min(z.min(), z_fake.min()), max(z.max(), z_fake.max()), 500)

fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle(f"PDF Estimation using GAN  (a_r={a_r}, b_r={b_r})")

axes[0].plot(z_vals, kde_real(z_vals), label="Real z", color="blue")
axes[0].plot(z_vals, kde_fake(z_vals), label="GAN p_h(z)", color="red", linestyle="--")
axes[0].set_title("KDE: Real vs Generated")
axes[0].set_xlabel("z")
axes[0].set_ylabel("density")
axes[0].legend()

axes[1].hist(z, bins=50, density=True, alpha=0.5, label="Real z", color="blue")
axes[1].hist(z_fake, bins=50, density=True, alpha=0.5, label="GAN z_f", color="red")
axes[1].set_title("Histogram Overlay")
axes[1].set_xlabel("z")
axes[1].set_ylabel("density")
axes[1].legend()

fig1.tight_layout()
fig1.savefig("pdf_estimation_gan.png", dpi=150)
print("pdf_estimation_gan.png saved")

fig2, ax = plt.subplots(figsize=(7, 5))
ax.plot(g_loss_list, label="Generator", color="red")
ax.plot(d_loss_list, label="Discriminator", color="blue")
ax.set_title("Training Losses")
ax.set_xlabel("Epoch")
ax.set_ylabel("BCE Loss")
ax.legend()
fig2.tight_layout()
fig2.savefig("training_loss.png", dpi=150)
print("training_loss.png saved")

plt.show()

print(f"\nreal z   mean={z.mean():.3f}  std={z.std():.3f}")
print(f"fake z_f mean={z_fake.mean():.3f}  std={z_fake.std():.3f}")
print(f"final D loss: {d_loss_list[-1]:.4f}  G loss: {g_loss_list[-1]:.4f}")

mean_diff = abs(z.mean() - z_fake.mean())
std_diff = abs(z.std() - z_fake.std())
final_avg_loss = (d_loss_list[-1] + g_loss_list[-1]) / 2

print(f"\nmean diff: {mean_diff:.3f}")
if mean_diff < 1.0:
    print("mode coverage ok - generated mean is close to real")
else:
    print("some mode shift, generated mean is off")

print(f"avg final loss: {final_avg_loss:.4f}  (ideal ~0.693)")
if abs(final_avg_loss - 0.693) < 0.15:
    print("training looks stable")
else:
    print("training may not have converged fully")

print(f"std diff: {std_diff:.3f}")
if std_diff < z.std() * 0.2:
    print("spread of generated samples is close to real")
else:
    print("spread differs, might need more epochs")

torch.save(G.state_dict(), "generator.pth")
print("generator weights saved")
