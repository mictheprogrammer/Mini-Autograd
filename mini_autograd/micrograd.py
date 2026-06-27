# %%
# 0

import math
import torch
import numpy as np
import random

# %%
# 1


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data = {self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __pow__(self, other):
        assert isinstance(other, (float, int))
        out = Value(self.data**other, (self,))

        def _backward():
            self.grad += out.grad * other * self.data ** (other - 1)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += out.grad * (1 - t**2)

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,))

        def _backward():
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# %%
# 2

x1 = Value(2.0)
x2 = Value(0.0)

w1 = Value(-3.0)
w2 = Value(1.0)
b = Value(6.7)

w1x1 = x1 * w1
w2x2 = x2 * w2

w1x1w2x2 = w1x1 + w2x2
n = w1x1w2x2 + b
e = (2 * n).exp()
o = (e - 1) / (e + 1)
# %%
# 3

o.grad
e.grad
n.grad
w1x1w2x2.grad
b.grad
w1x1.grad
w2x2.grad
x1.grad
x2.grad
w1.grad
w2.grad

# %%
# 4

e.grad = 0.0
n.grad = 0.0
w1x1w2x2.grad = 0.0
b.grad = 0.0
w1x1.grad = 0.0
w2x2.grad = 0.0
x1.grad = 0.0
x2.grad = 0.0
w1.grad = 0.0
w2.grad = 0.0


# %%
# 5

x1 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
x2 = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)

w1 = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
w2 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
b = torch.tensor([6.7], dtype=torch.float64, requires_grad=True)

n = x1 * w1 + x2 * w2 + b
o = torch.tanh(n)

n.retain_grad()
o.retain_grad()
o.backward()

assert (
    x1.grad is not None
    and x2.grad is not None
    and o.grad is not None
    and n.grad is not None
)
assert w1.grad is not None and w2.grad is not None and b.grad is not None

print("------")
print("o", o.grad.item())
print("n", n.grad.item())
print("x2", x2.grad.item())
print("w2", w2.grad.item())
print("x1", x1.grad.item())
print("w1", w1.grad.item())
print("b", b.grad.item())

# %%
# 6


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in (zip(self.w, x))), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        nins = [nin] + nouts[:-1]
        self.layers = [Layer(nin, nout) for nin, nout in zip(nins, nouts)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


n = MLP(3, [4, 4, 1])

# %%
# 7

xs = [[2.0, 3.0, -1], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

# %%
# 8

ypred = [n(x) for x in xs]
print(ypred)
MSE = sum((pred[0] - truth) ** 2 for pred, truth in zip(ypred, ys))
MSE.backward()
MSE

# %%
# 9
EPOCHS = 50
LR = 0.01

for epoch in range(EPOCHS):
    ypred = [n(x) for x in xs]
    print(ypred)
    MSE = sum((pred[0] - truth) ** 2 for pred, truth in zip(ypred, ys))
    MSE.backward()
    for p in n.parameters():
        p.data += -LR * p.grad
