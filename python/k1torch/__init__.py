"""k1torch: a tiny PyTorch-like framework for the SHIELD Tablet GPU (Tegra K1) over cuda-usb.

    import k1torch as kt
    from k1torch import nn, optim, functional as F
"""
from . import data, functional, nn, optim
from .tensor import (Tensor, device_name, empty, full, manual_seed, no_grad, randn, synchronize,
                     tensor, zeros)

__all__ = ["Tensor", "data", "device_name", "empty", "full", "functional", "manual_seed", "nn",
           "no_grad", "optim", "randn", "synchronize", "tensor", "zeros"]
