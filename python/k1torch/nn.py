"""Layers in the style of torch.nn."""
import math

from . import functional as F
from .tensor import Tensor, randn, tensor, zeros


class Parameter(Tensor):
    def __new__(cls, t):
        t.requires_grad = True
        t.__class__ = cls
        return t

    def __init__(self, t):
        pass


class Module:
    def __init__(self):
        self.training = True

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError

    def children(self):
        return [v for v in vars(self).values() if isinstance(v, Module)]

    def named_parameters(self, prefix=""):
        for name, v in vars(self).items():
            if isinstance(v, Parameter):
                yield prefix + name, v
            elif isinstance(v, Module):
                yield from v.named_parameters(prefix + name + ".")
            elif isinstance(v, list):
                for i, m in enumerate(v):
                    if isinstance(m, Module):
                        yield from m.named_parameters(f"{prefix}{name}.{i}.")

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def train(self, mode=True):
        self.training = mode
        for m in self.children():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def save(self, path):
        """Writes all parameters to a file (downloads them from the GPU)."""
        with open(path, "wb") as f:
            for name, p in self.named_parameters():
                f.write(p.to_array().tobytes())

    def load(self, path):
        with open(path, "rb") as f:
            for name, p in self.named_parameters():
                data = f.read(p.numel() * 4)
                p.storage = tensor(data, shape=p.shape).storage
                p.offset = 0

    def __repr__(self):
        inner = "".join(f"\n  ({i}): {m!r}" for i, m in enumerate(getattr(self, "layers", [])))
        return f"{type(self).__name__}({self.extra_repr()}{inner}{chr(10) if inner else ''})"

    def extra_repr(self):
        return ""


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = Parameter(randn(out_features, in_features, std=math.sqrt(2.0 / in_features)))
        self.bias = Parameter(zeros(out_features)) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return f"{self.in_features}, {self.out_features}"


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.args = (in_channels, out_channels, kernel_size, stride, padding)
        fan_in = in_channels * kernel_size * kernel_size
        self.weight = Parameter(randn(out_channels, in_channels, kernel_size, kernel_size,
                                      std=math.sqrt(2.0 / fan_in)))
        self.bias = Parameter(zeros(out_channels)) if bias else None
        self.stride, self.padding = stride, padding

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)

    def extra_repr(self):
        c_in, c_out, k, s, p = self.args
        return f"{c_in}, {c_out}, kernel_size={k}, stride={s}, padding={p}"


class ReLU(Module):
    def forward(self, x):
        return F.relu(x)


class MaxPool2d(Module):
    def __init__(self, kernel_size=2):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size)

    def extra_repr(self):
        return f"kernel_size={self.kernel_size}"


class Flatten(Module):
    def forward(self, x):
        return F.flatten(x)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def children(self):
        return self.layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CrossEntropyLoss(Module):
    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels)
