"""Differentiable operations. Each runs GPU kernels and records its backward pass."""
from ._driver import Ptr, device
from .tensor import Tensor, empty, grad_enabled, zeros

_NULL = Ptr(0)


def _track(out, inputs, backward):
    if grad_enabled() and any(t is not None and t.requires_grad for t in inputs):
        out._make_node(tuple(inputs), backward)
    return out


def _gemm(a, b, c, M, N, K, trans_a=0, trans_b=0, alpha=1.0, beta=0.0):
    device().launch("gemm", ((N + 15) // 16, (M + 15) // 16), (16, 16),
                    a.ptr, b.ptr, c.ptr, M, N, K, trans_a, trans_b, alpha, beta)


def relu(x):
    n = x.numel()
    y = empty(x.shape)
    device().launch1d("relu_forward", n, x.ptr, y.ptr, n)

    def backward(g):
        dx = empty(x.shape)
        device().launch1d("relu_backward", n, x.ptr, g.ptr, dx.ptr, n)
        return (dx,)

    return _track(y, (x,), backward)


def linear(x, weight, bias=None):
    """x [N, in] @ weight[out, in]^T + bias[out]"""
    rows, fin = x.shape
    fout = weight.shape[0]
    y = empty(rows, fout)
    _gemm(x, weight, y, rows, fout, fin, trans_b=1)
    if bias is not None:
        device().launch1d("add_bias_rows", rows * fout, y.ptr, bias.ptr, rows, fout)

    def backward(g):
        dx = dw = db = None
        if x.requires_grad:
            dx = empty(rows, fin)
            _gemm(g, weight, dx, rows, fin, fout)
        if weight.requires_grad:
            dw = empty(fout, fin)
            _gemm(g, x, dw, fout, fin, rows, trans_a=1)
        if bias is not None and bias.requires_grad:
            db = empty(fout)
            device().launch1d("bias_grad_rows", fout, g.ptr, db.ptr, rows, fout, 0)
        return dx, dw, db

    return _track(y, (x, weight, bias), backward)


def conv2d(x, weight, bias=None, stride=1, padding=0):
    """x [N, C, H, W], weight [OC, C, K, K] -> [N, OC, OH, OW]"""
    N, C, H, W = x.shape
    OC, C2, K, K2 = weight.shape
    assert C == C2 and K == K2, f"conv2d: input {x.shape} vs weight {weight.shape}"
    OH = (H + 2 * padding - K) // stride + 1
    OW = (W + 2 * padding - K) // stride + 1
    dims = (N, C, H, W, OC, K, stride, padding, OH, OW)
    y = empty(N, OC, OH, OW)
    device().launch1d("conv2d_forward", y.numel(), x.ptr, weight.ptr,
                      bias.ptr if bias is not None else _NULL, y.ptr, *dims)

    def backward(g):
        dx = dw = db = None
        if x.requires_grad:
            dx = empty(x.shape)
            device().launch1d("conv2d_backward_input", dx.numel(), g.ptr, weight.ptr, dx.ptr, *dims)
        if weight.requires_grad:
            dw = empty(weight.shape)
            device().launch("conv2d_backward_weight", weight.numel(), 128,
                            x.ptr, g.ptr, dw.ptr, *dims, 0)
        if bias is not None and bias.requires_grad:
            db = empty(OC)
            device().launch("conv2d_backward_bias", OC, 128, g.ptr, db.ptr, N, OC, OH * OW, 0)
        return dx, dw, db

    return _track(y, (x, weight, bias), backward)


def max_pool2d(x, kernel_size=2):
    N, C, H, W = x.shape
    K = kernel_size
    OH, OW = H // K, W // K
    y = empty(N, C, OH, OW)
    idx = empty(N, C, OH, OW, dtype="int32")
    device().launch1d("maxpool_forward", y.numel(), x.ptr, y.ptr, idx.ptr, N, C, H, W, K, OH, OW)

    def backward(g):
        dx = zeros(x.shape)
        device().launch1d("maxpool_backward", g.numel(), g.ptr, idx.ptr, dx.ptr, g.numel())
        return (dx,)

    return _track(y, (x,), backward)


def flatten(x):
    return x.view(x.shape[0], -1)


def cross_entropy(logits, labels):
    """Mean softmax cross-entropy. logits [N, C] float, labels [N] int32 -> scalar tensor."""
    N, C = logits.shape
    loss = zeros(1)
    grad = empty(N, C)  # computed in the same pass, used by backward
    device().launch1d("cross_entropy", N, logits.ptr, labels.ptr, grad.ptr, loss.ptr, N, C)

    def backward(g):
        device().launch1d("scale_by", grad.numel(), grad.ptr, g.ptr, grad.numel())
        return grad, None

    return _track(loss, (logits, labels), backward)


def count_correct(logits, labels, counter):
    """Adds the number of correct argmax predictions to counter (int32 [1]) on the GPU."""
    N, C = logits.shape
    device().launch1d("count_correct", N, logits.ptr, labels.ptr, counter.ptr, N, C)
