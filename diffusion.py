import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.utils import ModelEmaV3
from tqdm import tqdm


def forward_diffusion(x, T=40):
    ans = x.float() / 255
    beta = np.linspace(0.0001, 0.02, T)
    alpha = 1 - beta
    alpha_t = np.prod(alpha)
    noise = torch.randn_like(ans)
    ans = np.sqrt(alpha_t) * ans + np.sqrt(1 - alpha_t) * noise
    return (ans * 255).int()


class DDPM(nn.Module):
    def __init__(self, T=1000):
        super().__init__()
        self.beta = torch.linspace(0.0001, 0.02, T, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]


class TimeEmbedding(nn.Module):
    def __init__(self, steps, embed_dim):
        super().__init__()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pos = torch.arange(steps).unsqueeze(1).float()
        angle = pos * div
        self.embeddings = torch.zeros(steps, embed_dim, requires_grad=False)
        self.embeddings[:, 0::2] = torch.sin(angle)
        self.embeddings[:, 1::2] = torch.cos(angle)

    def forward(self, _, t):
        return self.embeddings[t][:, :, None, None]


class ResNetBlock(nn.Module):
    def __init__(self, num_chans, kernel_size, padding, num_groups, dropout):
        super().__init__()
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.conv1 = nn.Conv2d(num_chans, num_chans, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(num_chans, num_chans, kernel_size=kernel_size, padding=padding)
        self.group_norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=num_chans)
        self.group_norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=num_chans)

    def forward(self, x, embeddings):
        res = self.conv1(self.activation(self.group_norm1(x)))
        res = self.dropout(res)
        res = self.conv2(self.activation(self.group_norm2(res)))
        return res + x + embeddings[:, :x.shape[1], :, :]


class Attention(nn.Module):
    def __init__(self, num_chans, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.first_projection = nn.Linear(num_chans, num_chans * 3)
        self.second_projection = nn.Linear(num_chans, num_chans)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.first_projection(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        x = F.scaled_dot_product_attention(x[0], x[1], x[2], is_causal=False,
                                           dropout_p=self.dropout)
        h, w = x.shape[2:]
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.second_projection(x)
        return rearrange(x, 'b h w C -> b C h w')


class UnetLayer(nn.Module):
    def __init__(self, attention, num_groups, dropout, kernel_size, padding, num_heads,
                 num_chans, upscale_kernel_size=None):
        super().__init__()
        self.ResBlock1 = ResNetBlock(num_chans=num_chans, kernel_size=kernel_size, padding=padding,
                                     num_groups=num_groups, dropout=dropout)
        self.ResBlock2 = ResNetBlock(num_chans=num_chans, kernel_size=kernel_size, padding=padding,
                                     num_groups=num_groups, dropout=dropout)
        if upscale_kernel_size is not None:
            self.conv = nn.ConvTranspose2d(num_chans, num_chans // 2,
                                           kernel_size=upscale_kernel_size,
                                           stride=2, padding=padding)
        else:
            self.conv = nn.Conv2d(num_chans, num_chans * 2, kernel_size=kernel_size, stride=2,
                                  padding=padding)
        if attention:
            self.attention_layer = Attention(num_chans, num_heads=num_heads, dropout=dropout)
        else:
            self.attention_layer = None

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if self.attention_layer is not None:
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x


class UNET(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512, 512, 384), attentions=(False, True, False,
                                                                           False, False, True),
                 upscale=(None, None, None, 4, 4, 4), num_groups=32, dropout=0.1, num_heads=8,
                 kernel_size=3, padding=1, input_channels=1, output_channels=1, steps=1000):
        super().__init__()
        self.num_layers = len(channels)
        self.embeddings = TimeEmbedding(steps=steps, embed_dim=max(channels))
        self.activation = nn.ReLU(inplace=True)
        self.input_conv = nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1)
        out_channels = channels[-1] // 2 + channels[0]
        self.last_conv = nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels // 2, output_channels, kernel_size=1)
        layers = []
        for i in range(self.num_layers):
            layers.append(UnetLayer(attention=attentions[i], num_groups=num_groups,
                                    dropout=dropout, kernel_size=kernel_size, padding=padding,
                                    num_heads=num_heads, num_chans=channels[i],
                                    upscale_kernel_size=upscale[i]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, t):
        x = self.input_conv(x)
        residuals = []
        num_downsampling_layers = self.num_layers // 2
        for i in range(num_downsampling_layers):
            x, r = self.layers[i](x, self.embeddings(x, t))
            residuals.append(r)
        for i in range(num_downsampling_layers, self.num_layers):
            x = torch.concat((self.layers[i](x, self.embeddings(x, t))[0],
                              residuals[self.num_layers - i - 1]), dim=1)
        return self.output_conv(self.activation(self.last_conv(x)))


def train(batch_size=64, steps=1000, epochs=15, ema_decay=0.9999, lr=2e-5):
    train_set = datasets.MNIST(root='./data', train=True, download=False,
                               transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=2)
    ddpm = DDPM(T=steps)
    model = UNET()
    ema = ModelEmaV3(model, decay=ema_decay)
    adam = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')
    for i in range(epochs):
        loss = 0
        for _, (x, _) in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{epochs}")):
            x = F.pad(x, (2,) * 4)
            t = torch.randint(0, steps, (batch_size,))
            e = torch.randn_like(x, requires_grad=False)
            a = ddpm.alpha[t].view(batch_size, 1, 1, 1)
            x = torch.sqrt(a) * x + torch.sqrt(1 - a) * e
            output = model(x, t)
            adam.zero_grad()
            loss = criterion(output, e)
            loss += loss.item()
            loss.backward()
            adam.step()
            ema.update(model)
        print(f'loss for epoch {i+1}: {loss / (60000/batch_size):.6f}')

    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': adam.state_dict(),
        'ema': ema.state_dict()
    }
    torch.save(checkpoint, 'checkpoint')


def generate(checkpoint, steps=1000, ema_decay=0.9999):
    model = UNET()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM(T=steps)
    times = [0, 50, 100, 300, 500, 700, 999]
    images = []
    with torch.no_grad():
        model = ema.module.eval()
        for i in range(10):
            z = torch.randn(1, 1, 32, 32)
            for t in reversed(range(1, steps)):
                t = [t]
                temp = scheduler.beta[t] / torch.sqrt(1 - scheduler.alpha[t]) * torch.sqrt(
                    1 - scheduler.beta[t])
                z = z / torch.sqrt(1 - scheduler.beta[t]) - temp * model(z, t)
                if t[0] in times:
                    images.append(z)
                e = torch.randn(1, 1, 32, 32)
                z = z + e * torch.sqrt(scheduler.beta[t])
            temp = scheduler.beta[0] / torch.sqrt(1 - scheduler.alpha[0]) * torch.sqrt(
                1 - scheduler.beta[0])
            x = z / torch.sqrt(1 - scheduler.beta[0]) - temp * model(z, [0])
            images.append(x)
            x = rearrange(x.squeeze(0), 'c h w -> h w c').detach()
            x = x.numpy()
            plt.imshow(x)
            plt.show()
            images = []


if __name__ == '__main__':
    # img = cv2.imread('IMG_4522.JPG')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # min_ax = min(img.shape[:2])
    # size = 128
    # diffused = forward_diffusion(torch.from_numpy(cv2.resize(img, (size, size))))
    # plt.imshow(diffused)
    # plt.show()
    train(lr=2e-5, epochs=75)
    generate(torch.load("checkpoint"))
