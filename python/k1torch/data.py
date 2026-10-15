"""Batching for datasets that already live on the GPU (no USB traffic per batch)."""
import random

from ._driver import device
from .tensor import empty, tensor


class DataLoader:
    """Iterates (x, y) batches of GPU tensors. With shuffle, a new permutation is uploaded
    once per epoch and batches are gathered on the GPU."""

    def __init__(self, x, y, batch_size=64, shuffle=False, drop_last=True):
        assert len(x) == len(y)
        self.x, self.y = x, y
        self.batch_size, self.shuffle, self.drop_last = batch_size, shuffle, drop_last

    def __len__(self):
        n = len(self.x)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n, bs = len(self.x), self.batch_size
        if not self.shuffle:
            for i in range(len(self)):
                yield self.x[i * bs:(i + 1) * bs], self.y[i * bs:(i + 1) * bs]
            return
        order = list(range(n))
        random.shuffle(order)
        perm = tensor(order, dtype="int32")
        row = self.x.numel() // n
        for i in range(len(self)):
            idx = perm[i * bs:(i + 1) * bs]
            m = len(idx)
            xb = empty((m,) + self.x.shape[1:])
            yb = empty(m, dtype="int32")
            device().launch1d("gather_rows", m * row, self.x.ptr, idx.ptr, xb.ptr, m, row)
            device().launch1d("gather_int", m, self.y.ptr, idx.ptr, yb.ptr, m)
            yield xb, yb
