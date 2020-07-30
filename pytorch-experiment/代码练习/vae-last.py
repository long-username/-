'''
1. forget zero_grad
2. why the value become < 0 
3. 出现图片变白
4. Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
5. s是么时候 [0, 1] [-1, 1]?? 出现 3, 4 问题
6. dis_loss = nan (梯度消失??)
7. cross_entrpy 限制 [0-1] 问题 输出限制sigmoid函数
8. con_loss = nan # con_loss = mse(out, pics) + 0.5 * tch.sum(tch.exp(v) - 1 - v + m.pow(2))
    |------在于 size_average 上 ???


出现的 loss = nan 的 问题
1. 网络参数太多 导致无法优化
2. log x (x <= 0)
3. randn_like 写成 rand_like
    原因在于rand_like 之后 eps * exp(v) 之后 导数非常的大 之后的 以至于 inf -> nan
4. mse(average_size=False) ?? 无碍
'''

'''
code = eps * v ** 2 + m
loss = mse(out, y) + v ** 4 + 4 * tchlog(tch.abs(v)) + m ** 4

这个loss 说明 e^x 正数变换是一个非常优秀的变换
'''

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

        self.fc1 = nn.Linear(self.size, 400)
        # self.fc2 = nn.Linear(678, 256)
        # self.fc3 = nn.Linear(256, 128)

        self.codeM = nn.Linear(400, code_dim)
        self.codeV = nn.Linear(400, code_dim)

        self.fc4 = nn.Linear(code_dim, 400)
        # self.fc5 = nn.Linear(256, 678)
        self.fc6 = nn.Linear(400, self.size)
        self.normal_d = tch.distributions.normal.Normal(0, 1)

    def forward(self, x):

        code, v, m = self.encoders(x)
        x = self.decoders(code)

        return x

    def decoders(self, code):

        x = ReLU(self.fc4(code))
        # x = ReLU(self.fc5(x))
        x = F.sigmoid(self.fc6(x))

        return x
        
    def encoders(self, x):

        x = ReLU(self.fc1(x))
        # x = ReLU(self.fc2(x))
        # x = ReLU(self.fc3(x))

        v = self.codeV(x)
        m = self.codeM(x)

        code = self.reparameterize(m, v)
        
        return code, v, m

    def reparameterize(self, mu, logvar):
        if True:
            std = tch.exp(0.5*logvar)
            eps = tch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    KLD = -0.5 * tch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


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

for j in range(epoch):
    con = 0
    dis = 0
    for i, data in enumerate(train):

        pics, _ = data
        sgd.zero_grad()
        pics = pics.view((-1, 784))

        code, v, m = vae.encoders(pics)
        out = vae.decoders(code)

        con_loss = mse(out, pics) + 0.5 * tch.sum(tch.exp(v) - 1 - v + m.pow(2))
        # con_loss = loss_function(out, pics, m, v)
        con_loss.backward()

        con += con_loss.item()

        sgd.step()

        if i % 20 == 19:
            print(f'content loss is {con / 20}')
            con = 0
        if i % 100 == 99:
            noise = vae.normal_d.sample((8, 12))
            fpi = vae.decoders(noise)
            fpi = fpi.view(-1, 1, 28, 28)
            fpi = tchv.utils.make_grid(fpi)

            # print(fpi.shape)
            fpi = fpi.detach().numpy()
            fpi = np.transpose(fpi, (1, 2, 0))
            plt.imshow(fpi)
            plt.savefig(f'./Desktop/sc/scc/{j}-{i}.png')
            plt.close()

