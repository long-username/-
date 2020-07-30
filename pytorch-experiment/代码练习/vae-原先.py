import torch as tch
from torch.nn import Module
from torch import nn
import numpy as np
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import RMSprop
from matplotlib import pyplot as plt
import torchvision as tchv
ReLU = F.relu
torch = tch

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))

        m, v = self.fc21(h1), self.fc22(h1)
        code = self.reparameterize(m, v)
        return code, m, v

    def reparameterize(self, mu, logvar):
        if True:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        code, mu, logvar = self.encode(x.view(-1, 784))
        return self.decode(code), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

tr = transforms.Compose(
    [transforms.ToTensor()])

train = MNIST('./data', transform=tr, download=True)
test = MNIST('./data', transform=tr, train=False, download=True)

epoch = 10
batch_size = 30
train = DataLoader(train, batch_size, True)
test = DataLoader(test, batch_size, False)
vae = VAE()
sgd = tch.optim.RMSprop(vae.parameters())
mse = nn.MSELoss()

for j in range(epoch):
    con = 0
    dis = 0
    for i, data in enumerate(train):

        pics, _ = data
        sgd.zero_grad()
        pics = pics.view((-1, 784))

        code, m, v = vae.forward(pics)

        # dis_loss = 0.5 * tch.sum(v.exp() - v.add(1) + m.pow(2))
        # print(dis_loss)

        # con_loss = mse(out, pics) + dis_loss
        con_loss = loss_function(code, pics, m, v)
        con_loss.backward()

        con += con_loss.item()

        sgd.step()

        if i % 20 == 19:
            print(f'content loss is {con / 20}')
            con = 0
        if i % 100 == 99:
            noise = tch.randn_like(m)
            fpi= vae.decode(noise)
            fpi = fpi.view(-1, 1, 28, 28)
            fpi = tchv.utils.make_grid(fpi)

            # print(fpi.shape)
            fpi = fpi.detach().numpy()
            fpi = np.transpose(fpi, (1, 2, 0))
            plt.imshow(fpi)
            plt.savefig(f'./Desktop/sc/scc/{j}-{i}.png')
            plt.close()

