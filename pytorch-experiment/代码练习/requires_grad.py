'''
1. requires_grad 是有顺序的, 得在下一个运算前定义好 = True
2. requires_grad 在手动创建tensor时, 默认是 False
3. requires_grad 来自fn时, 默认继承来自运算的叶子结点的 requires_grad 属性
4. detach() 函数用来分离图变量
5. tch.nograd 并不改变以前的 requires_grad属性, 而是在域内的运算requires_grad属性不再继承,一律为False
6. 
'''
import torch as tch

v = tch.tensor([1])


w = v ** 2
c = w.detach()
# w.requires_grad = True

r = w * 2
v.requires_grad = True
r.backward()
with tch.no_grad():
    print((v ** 2).requires_grad)

print((v ** 2).requires_grad)

print(v.grad)