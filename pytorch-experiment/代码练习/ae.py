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



class VAE(Module):
    def __init__(self, size, code_dim):

        super(VAE, self).__init__()
        self.c, self.h, self.w = size
        self.size = self.c * self.h * self.w
        self.last_ne = 8

        self.fc1 = nn.Linear(self.size, 678)
        self.fc2 = nn.Linear(678, 256)
        self.fc3 = nn.Linear(256, code_dim)

        self.fc4 = nn.Linear(code_dim, 256)
        self.fc5 = nn.Linear(256, 678)
        self.fc6 = nn.Linear(678, self.size)
        self.normal_d = tch.distributions.normal.Normal(0, 1)

    def forward(self, x):

        code = self.encoders(x)
        x = self.decoders(code)

        return x

    def decoders(self, code):

        x = ReLU(self.fc4(code))
        x = ReLU(self.fc5(x))
        x = self.fc6(x)

        return x
        
    def encoders(self, x):

        x = ReLU(self.fc1(x))
        x = ReLU(self.fc2(x))
        code = ReLU(self.fc3(x))
        
        return code


tr = transforms.Compose(
    [transforms.ToTensor()])

train = MNIST('./data', transform=tr, download=True)
test = MNIST('./data', transform=tr, train=False, download=True)

epoch = 10
batch_size = 30
train = DataLoader(train, batch_size, True)
test = DataLoader(test, batch_size, False)
vae = VAE((1, 28, 28), 120)
sgd = tch.optim.RMSprop(vae.parameters())
mse = nn.MSELoss()
# dev = tch.device('cuda:0')
# vae.to(dev)

for j in range(epoch):
    con = 0
    dis = 0
    for i, data in enumerate(train):

        pics, _ = data
        # pics = pics.to(dev)
        sgd.zero_grad()
        pics = pics.view((-1, 784))

        out = vae.encoders(pics)
        out = vae.decoders(out)

        con_loss = mse(out, pics)
        con_loss.backward()

        con += con_loss.item()

        sgd.step()

        if i % 20 == 19:
            print(f'content loss is {con / 20}')
            con = 0
        if i % 100 == 99:
            # noise = vae.normal_d.sample((8, 3))
            fpi = vae.forward(pics)
            fpi = fpi.view(-1, 1, 28, 28)
            fpi = tchv.utils.make_grid(fpi)

            fpi = fpi.detach().cpu().numpy()
            fpi = np.transpose(fpi, (1, 2, 0))
            plt.imshow(fpi)
            plt.savefig(f'./Desktop/sc/{j}-{i}.png')
            plt.close()

