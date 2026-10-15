"""Gradient check: analytic gradients from k1torch vs. central finite differences.
Run with the tablet connected: python3 tests/test_grad.py"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import k1torch as kt  # noqa: E402
from k1torch import functional as F  # noqa: E402

EPS = 1e-2
ATOL, RTOL = 2e-3, 2e-2  # float32 + fast __expf/__logf make finite differences noisy near 0
SAMPLES = 12  # checked elements per input


def check(name, loss_fn, inputs):
    """loss_fn(*inputs) -> scalar tensor. Compares d loss / d input for every input."""
    for t in inputs:
        t.grad = None
    loss_fn(*inputs).backward()
    worst = 0.0
    for k, t in enumerate(inputs):
        analytic = t.grad.to_array()
        values = t.to_array()
        for i in random.sample(range(len(values)), min(SAMPLES, len(values))):
            losses = []
            for delta in (EPS, -EPS):
                shifted = values[:]
                shifted[i] += delta
                args = list(inputs)
                args[k] = kt.tensor(shifted, shape=t.shape)
                with kt.no_grad():
                    losses.append(loss_fn(*args).item())
            numeric = (losses[0] - losses[1]) / (2 * EPS)
            worst = max(worst, abs(numeric - analytic[i]) / (ATOL + RTOL * abs(numeric)))
    ok = worst <= 1.0
    print(f"  {'OK ' if ok else 'BAD'} {name:<10} error / tolerance = {worst:.2f}")
    return ok


def main():
    kt.manual_seed(0)
    random.seed(0)
    print("gradient check on", kt.device_name())

    def param(*shape):
        return kt.randn(*shape, requires_grad=True)

    labels = kt.tensor([1, 0, 3], dtype="int32")
    head1 = kt.randn(4, 3 * 5 * 5)  # fixed linear heads turn feature maps into 4 logits
    head2 = kt.randn(4, 3 * 3 * 3)
    head3 = kt.randn(4, 2 * 2 * 2)

    def logits(features, head):
        return F.linear(F.flatten(features), head)

    results = [
        check("linear", lambda x, w, b: F.cross_entropy(F.linear(x, w, b), labels),
              [param(3, 5), param(4, 5), param(4)]),
        check("relu", lambda x: F.cross_entropy(F.relu(x), labels), [param(3, 4)]),
        check("conv2d", lambda x, w, b: F.cross_entropy(logits(F.conv2d(x, w, b, 1, 1), head1), labels),
              [param(3, 2, 5, 5), param(3, 2, 3, 3), param(3)]),
        check("conv2d/s2", lambda x, w: F.cross_entropy(logits(F.conv2d(x, w, None, 2, 0), head2), labels),
              [param(3, 2, 6, 6), param(3, 2, 2, 2)]),
        check("maxpool", lambda x: F.cross_entropy(logits(F.max_pool2d(x, 2), head3), labels),
              [param(3, 2, 4, 4)]),
    ]
    print("all gradients OK" if all(results) else "SOME GRADIENTS ARE WRONG")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
