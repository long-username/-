'''
pytorch 程序
CNN 的程序

1. 定义CNN 结构(继承nn.Module)
    a. __init__ 别忘了 super(cnn, self).__init__()
    b. 定义 forward 前向传播
    c.
2. load 数据
    a. 使用现有数据库 torchvision.datasets
    b. 制作torch.datasets 数据库对象
    c. 读入数据 torch.untils.DataLoader
    d. 要是显示,利用np.transpose函数 将(c, h, w) -> (h, w, c)
    e. datasets 对象是一个可遍历的对象 enumerate(loader)
2. 优化过程
    a. 定义CNN对象
    b. 各类参数- batch epoch
    c. 在nn.optim 模块 例如SGD(CNN.parameters(), lr, ...)传入
    d. for(epoch) ..for(enumerate(loader))..
    e. 要有loss 在nn 模块 
    f. 
'''


import torch as tch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from torch.optim import SGD

batch = 20


class cnn(nn.Module):
    def __init__(self, batch):
        super(cnn, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)

        # what is middle bewteen fc and conv?
        self.fc1 = nn.Linear(32 * 28 * 28, 90)
        self.fc2 = nn.Linear(90, 10)

        self.batch = batch

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(self.batch, -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train = datasets.MNIST(
    './data', train=True, download=True, transform=transform)
trainloader = DataLoader(train, batch_size=batch, shuffle=True)

test = datasets.MNIST(
    './data', train=True, download=False, transform=transform)
testloader = DataLoader(test, batch_size=batch, shuffle=False)

# it = iter(testloader)
# a, _ = it.next()
# pics = make_grid(a)
# pics /= 2
# pics += 0.5
# pics = np.transpose(pics, (1, 2, 0))
# plt.imshow(pics)
# plt.show()

n = cnn(batch)
device = tch.device("cuda:0" if tch.cuda.is_available() else "cpu")
n.to(device)
sgd = SGD(n.parameters(), lr=0.001, momentum=0.9)
ce = nn.CrossEntropyLoss()
epoch = 10
for i in range(epoch):

    r_loss = 0
    for index, data in enumerate(trainloader):
       
        inp, lab = data
        inp = inp.to(device)
        lab = lab.to(device)
        sgd.zero_grad()
        out = n(inp)
        loss = ce(out, lab)
        loss.backward()
        sgd.step()

        r_loss += loss.item()
        if index % 200 == 199:
            print(f'{(i + 1) * index} time loss is {r_loss / 1999}')
            r_loss = 0


print('ok')

counter = 0
total = 0
for i, data in enumerate(testloader):

    inp, lab = data
    inp = inp.to(device)
    lab = lab.to(device)
    out = n(inp)
    out = tch.argmax(out, 1)
    total += len(lab)
    counter += (out == lab).sum().item()
    
acc = counter/total
print(acc)


# scatter_ ??
