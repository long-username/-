'''
这个是变量的调用链 grad_fn.next_functions
只有 requires_grad = True 之后才会记录
((<AccumulateGrad object at 0x000001DE15650160>, 0), (<AddBackward1 object at 0x000001DE15650588>, 0))
后面的 0 不知道什么意思
'''
import torch as tch

def add(a, b):
    return a + b

a = tch.tensor([1])
b = tch.tensor([1])
a.requires_grad_(True)
b.requires_grad_(True)

c = b ** 2
d = c * 2

e = a + add(d * 2, a)
e.backward()
print(e.grad_fn.next_functions)