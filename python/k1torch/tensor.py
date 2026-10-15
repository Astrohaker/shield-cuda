"""GPU tensors (float32 / int32, contiguous) with reverse-mode autograd."""
import random
from array import array

from ._driver import Ptr, device, float_bits

_grad_enabled = True


class no_grad:
    """Context manager: operations inside do not record the autograd graph."""

    def __enter__(self):
        global _grad_enabled
        self._prev, _grad_enabled = _grad_enabled, False

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self._prev


def grad_enabled():
    return _grad_enabled


class _Storage:
    __slots__ = ("ptr", "size")

    def __init__(self, nbytes):
        self.ptr, self.size = device().malloc(max(nbytes, 4))

    def __del__(self):
        try:
            device().free(self.ptr, self.size)
        except Exception:  # interpreter shutdown
            pass


def _prod(shape):
    n = 1
    for s in shape:
        n *= s
    return n


class Tensor:
    def __init__(self, shape, dtype="float32", storage=None, offset=0, requires_grad=False):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.storage = storage or _Storage(_prod(self.shape) * 4)
        self.offset = offset  # in elements
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = None  # fn(grad) for non-leaf tensors
        self._parents = ()

    # -- basics -------------------------------------------------------------------------------

    @property
    def ptr(self):
        return Ptr(self.storage.ptr + self.offset * 4)

    def numel(self):
        return _prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def __repr__(self):
        grad = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={self.shape}, dtype={self.dtype}{grad})"

    def __len__(self):
        return self.shape[0]

    def to_array(self):
        data = device().download(self.ptr.value, self.numel() * 4)
        a = array("f" if self.dtype == "float32" else "i")
        a.frombytes(data)
        return a

    def tolist(self):
        flat = self.to_array().tolist()
        for dim in reversed(self.shape[1:]):
            flat = [flat[i:i + dim] for i in range(0, len(flat), dim)]
        return flat

    def item(self):
        assert self.numel() == 1, "item() needs a single-element tensor"
        return self.to_array()[0]

    def detach(self):
        return Tensor(self.shape, self.dtype, self.storage, self.offset)

    # -- views --------------------------------------------------------------------------------

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if -1 in shape:
            known = _prod(s for s in shape if s != -1)
            shape = tuple(self.numel() // known if s == -1 else s for s in shape)
        assert _prod(shape) == self.numel(), f"cannot view {self.shape} as {shape}"
        out = Tensor(shape, self.dtype, self.storage, self.offset)
        if _grad_enabled and self.requires_grad:
            src_shape = self.shape
            out._make_node((self,), lambda g: (g.view(src_shape),))
        return out

    reshape = view

    def __getitem__(self, index):
        """Slicing along the first dimension (contiguous), e.g. x[64:128]. No autograd."""
        if isinstance(index, int):
            index = slice(index, index + 1)
        start, stop, step = index.indices(self.shape[0])
        assert step == 1, "only contiguous slices are supported"
        row = _prod(self.shape[1:])
        return Tensor((stop - start,) + self.shape[1:], self.dtype, self.storage,
                      self.offset + start * row)

    # -- in-place helpers (no autograd) ---------------------------------------------------------

    def fill_(self, value):
        if self.dtype == "int32":
            device().memset32(self.ptr.value, int(value) & 0xFFFFFFFF, self.numel())
        else:
            device().memset32(self.ptr.value, float_bits(float(value)), self.numel())
        return self

    def zero_(self):
        return self.fill_(0)

    def add_(self, other, alpha=1.0):
        assert other.numel() == self.numel()
        device().launch1d("axpy", self.numel(), self.ptr, other.ptr, float(alpha), self.numel())
        return self

    def __iadd__(self, other):
        return self.add_(other)

    # -- autograd -----------------------------------------------------------------------------

    def _make_node(self, parents, backward):
        self.requires_grad = True
        self._parents = parents
        self._backward = backward

    def backward(self, grad=None):
        if grad is None:
            assert self.numel() == 1, "backward() without a gradient needs a scalar"
            grad = full(self.shape, 1.0)
        order, seen = [], set()

        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                if p is not None:  # optional inputs, e.g. a missing bias
                    visit(p)
            order.append(t)

        visit(self)
        grads = {id(self): grad}
        for t in reversed(order):
            g = grads.pop(id(t), None)
            if g is None:
                continue
            if t._backward is None:  # leaf: accumulate into .grad
                if t.grad is None:
                    t.grad = g
                else:
                    t.grad.add_(g)
                continue
            for parent, pg in zip(t._parents, t._backward(g)):
                if pg is None or not parent.requires_grad:
                    continue
                if id(parent) in grads:
                    grads[id(parent)].add_(pg)
                else:
                    grads[id(parent)] = pg
            t._backward, t._parents = None, ()  # free the graph as we go


# -- factories ------------------------------------------------------------------------------------

def empty(*shape, dtype="float32", requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(shape, dtype, requires_grad=requires_grad)


def full(shape, value, dtype="float32", requires_grad=False):
    return empty(shape, dtype=dtype, requires_grad=requires_grad).fill_(value)


def zeros(*shape, dtype="float32", requires_grad=False):
    t = empty(*shape, dtype=dtype, requires_grad=requires_grad)
    return t.zero_()


def _flatten(data):
    """Nested lists -> (shape, flat list)."""
    shape, d = [], data
    while isinstance(d, (list, tuple)):
        shape.append(len(d))
        d = d[0] if d else None
    if not shape:
        return [], [data]
    flat = data
    for _ in range(len(shape) - 1):
        flat = [x for sub in flat for x in sub]
    return shape, list(flat)


def tensor(data, dtype="float32", shape=None, requires_grad=False):
    """Creates a GPU tensor from a nested list, an array.array or bytes (one upload)."""
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
        assert shape is not None, "shape is required for raw bytes"
    elif isinstance(data, array):
        raw = data.tobytes()
        shape = shape or (len(data),)
    else:
        nested_shape, flat = _flatten(data)
        shape = shape or tuple(nested_shape)
        raw = array("f" if dtype == "float32" else "i", flat).tobytes()
    t = Tensor(shape, dtype, requires_grad=requires_grad)
    assert len(raw) == t.numel() * 4, f"{len(raw) // 4} values do not fit shape {t.shape}"
    device().upload(t.ptr.value, raw)
    return t


def randn(*shape, std=1.0, requires_grad=False):
    """Normal random tensor (generated on the host: used for weight init)."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    values = array("f", (random.gauss(0.0, std) for _ in range(_prod(shape))))
    return tensor(values, shape=shape, requires_grad=requires_grad)


def manual_seed(seed):
    random.seed(seed)


def synchronize():
    device().synchronize()


def device_name():
    return device().name
