"""
ADAPTED FROM https://github.com/tcapelle/Diffusion-Models-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.notebook import get_tqdm
tqdm = get_tqdm()


class Diffusion:

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, image_size=64, nc=3, device="cpu"):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.nc = nc
        self.device = device

        self.beta = torch.linspace(self.beta_start, self.beta_end, self.T).to(self.device)
        self.alpha = (1 - self.beta).to(self.device)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(self.device)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.T, size=(n,))

    def sample(self, model, n=None, x=None, intern_noise=None, prt=True):
        if prt:
            print(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            if x is None:
                x = torch.randn((n, self.nc, self.image_size, self.image_size)).to(self.device)
            else:
                n = x.shape[0]
            steps = reversed(range(1, self.T))
            if prt:
                steps = tqdm(steps, total=self.T)
            for i in steps:
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    if intern_noise is None:
                        noise = torch.randn_like(x)
                    else:
                        noise = intern_noise[i]
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        return x


class UNet(nn.Module):
    def __init__(self, nc=3, time_dim=256, image_size=64, deep_conv=False, size=8, device="cpu"):
        super(UNet, self).__init__()
        self.time_dim = time_dim
        self.device = device
        self.inc = DoubleConv(nc, size*8)
        self.down1 = Down(size*8, size*16)
        self.sa1 = SelfAttention(size*16, image_size // 2)
        self.down2 = Down(size*16, size*32)
        self.sa2 = SelfAttention(size*32, image_size // 4)
        self.down3 = Down(size*32, size*32)
        self.sa3 = SelfAttention(size*32, image_size // 8)

        if deep_conv:
            self.bot = nn.Sequential(
                DoubleConv(size*32, size*64),
                DoubleConv(size*64, size*64),
                DoubleConv(size*64, size*32)
            )
        else:
            self.bot = nn.Sequential(
                DoubleConv(size*32, size*32),
                DoubleConv(size*32, size*32)
            )

        self.up1 = Up(size*64, size*16)
        self.sa4 = SelfAttention(size*16, image_size // 4)
        self.up2 = Up(size*32, size*8)
        self.sa5 = SelfAttention(size*8, image_size // 2)
        self.up3 = Up(size*16, size*8)
        self.sa6 = SelfAttention(size*8, image_size)
        self.outc = nn.Conv2d(size*8, nc, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1 / 10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super(DoubleConv, self).__init__()
        self.residual = residual
        if mid_channels is None:
            mid_channels = (in_channels + out_channels) // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")  # , align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


if __name__ == "__main__":
    import torch
    model = UNet(1, 3, 256, 3)
    print(model)
    print(model(torch.randn(1, 3, 16, 16), torch.tensor([2])))
