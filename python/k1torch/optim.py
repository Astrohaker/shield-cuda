"""Optimizers. Updates run on the GPU; nothing is read back."""
from ._driver import device
from .tensor import zeros


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params = list(params)
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self.velocity = [zeros(p.shape) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            n = p.numel()
            device().launch1d("sgd_step", n, p.ptr, p.grad.ptr, v.ptr, float(self.lr),
                              float(self.momentum), float(self.weight_decay), n)
