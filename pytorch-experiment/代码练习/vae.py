import torch as tch
from torch.nn import Module
from torch import nn
import numpy as np
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from matplotlib import pyplot as plt
import torchvision as tchv
ReLU = F.relu


class VAE(Module):
    def __init__(self, size, code_dim):

        super(VAE, self).__init__()
        self.c, self.h, self.w = size
        self.size = self.c * self.h * self.w
        self.last_ne = 200

        self.conv1 = nn.Linear(28*28, 400)
        self.conv3 = nn.Linear(400, self.last_ne)

        self.codeM = nn.Linear(self.last_ne, code_dim)
        self.codeV = nn.Linear(self.last_ne, code_dim)
        self.normal_d = tch.distributions.normal.Normal(0, 1)

        self.fconv3 = nn.Linear(code_dim, self.last_ne)
        self.fconv1 = nn.Linear(self.last_ne, 28 * 28)

    def forward(self, x):

        code, _, _ = self.encoder(x)
        x = self.decoder(code)

        return x

    def decoder(self, x):

        x = ReLU(self.fconv3(x))
        x = F.sigmoid(self.fconv1(x))

        return x

    def encoder(self, x):

        x = ReLU(self.conv1(x))
        x = ReLU(self.conv3(x))

        M = self.codeM(x)
        V = self.codeV(x)

        b, n = V.shape
        nda = self.normal_d.sample((b, n))
        code = M + tch.exp(V) * nda

        return code, V, M


tr = transforms.Compose(
    [transforms.ToTensor()])

train = MNIST('./data', transform=tr, download=True)
test = MNIST('./data', transform=tr, train=False, download=True)

epoch = 10
batch_size = 30
train = DataLoader(train, batch_size, True)
test = DataLoader(test, batch_size, False)
vae = VAE((1, 28, 28), 12)
sgd = tch.optim.RMSprop(vae.parameters())
mse = nn.MSELoss(size_average=False)

for i in range(epoch):
    con = 0
    dis = 0
    for i, data in enumerate(train):

        pics, _ = data
        pics = pics.view(-1, 28*28)
        sgd.zero_grad()
        code, v, m = vae.encoder(pics)
        out = vae.decoder(code)
        
        dis_loss = tch.exp(v) - (1 + v) + m ** 2
        dis_loss = dis_loss.sum()

        con_loss = mse(out, pics) + dis_loss
        con_loss.backward()

        con += con_loss.item()

        sgd.step()

        if i % 20 == 19:
            print(f'content loss is {con / 20}')
            con = 0
        if i % 100 == 99:
            noise = vae.normal_d.sample((8, 12))
            fpi = vae.decoder(noise)
            fpi = fpi.view((-1, 1, 28, 28))
            fpi = tchv.utils.make_grid(fpi)

            print(fpi.shape)

            fpi = fpi.detach().numpy()
            fpi = np.transpose(fpi, (1, 2, 0))
            plt.imshow(fpi)
            plt.savefig(f'./Desktop/pytorch-experiment/img/{i}.png')
            plt.close()

