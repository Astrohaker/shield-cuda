"""Trains a small CNN on a synthetic shape dataset on the SHIELD Tablet GPU.

Five classes of 20x20 noisy images: horizontal line, vertical line, two diagonals and a hollow
square, at random positions, sizes and brightness. The dataset is uploaded to the GPU once;
after that each training step only sends kernel launches over USB, and losses / accuracy are
accumulated on the GPU and read back once per epoch.

    source ~/cuda-usb/scripts/env.sh
    python3 ~/cuda-usb/python/examples/train_synthetic.py --epochs 5
"""
import argparse
import os
import random
import sys
import time
from array import array

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import k1torch as kt  # noqa: E402
from k1torch import functional as F, nn, optim  # noqa: E402
from k1torch.data import DataLoader  # noqa: E402

SIZE = 20
CLASSES = ["horizontal", "vertical", "diagonal \\", "diagonal /", "square"]


def make_image(label, rng):
    img = [0.0] * (SIZE * SIZE)
    level = rng.uniform(0.6, 1.0)

    def put(r, c):
        if 0 <= r < SIZE and 0 <= c < SIZE:
            img[r * SIZE + c] = level

    length = rng.randint(8, 16)
    r0, c0 = rng.randint(0, SIZE - length), rng.randint(0, SIZE - length)
    if label == 0:
        r = rng.randint(2, SIZE - 3)
        for c in range(c0, c0 + length):
            put(r, c)
    elif label == 1:
        c = rng.randint(2, SIZE - 3)
        for r in range(r0, r0 + length):
            put(r, c)
    elif label == 2:
        for k in range(length):
            put(r0 + k, c0 + k)
    elif label == 3:
        for k in range(length):
            put(r0 + k, c0 + length - 1 - k)
    else:
        side = rng.randint(6, 12)
        r0, c0 = rng.randint(0, SIZE - side), rng.randint(0, SIZE - side)
        for k in range(side):
            put(r0, c0 + k)
            put(r0 + side - 1, c0 + k)
            put(r0 + k, c0)
            put(r0 + k, c0 + side - 1)
    return [v + rng.gauss(0.0, 0.3) for v in img]


def make_dataset(n, seed):
    rng = random.Random(seed)
    labels = [rng.randrange(len(CLASSES)) for _ in range(n)]
    pixels = array("f")
    for label in labels:
        pixels.extend(make_image(label, rng))
    return pixels, array("i", labels)


def ascii_image(values):
    shades = " .:-=+*#%@"
    rows = []
    for r in range(SIZE):
        row = values[r * SIZE:(r + 1) * SIZE]
        rows.append("".join(shades[max(0, min(9, int(v * 9)))] for v in row))
    return rows


def evaluate(model, loader):
    correct = kt.zeros(1, dtype="int32")
    with kt.no_grad():
        for x, y in loader:
            F.count_correct(model(x), y, correct)
    return correct.item() / (len(loader) * loader.batch_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--train", type=int, default=6400)
    parser.add_argument("--test", type=int, default=1280)
    args = parser.parse_args()

    kt.manual_seed(0)
    print(f"device: {kt.device_name()}")

    t0 = time.time()
    train_x, train_y = make_dataset(args.train, seed=1)
    test_x, test_y = make_dataset(args.test, seed=2)
    t1 = time.time()
    shape = (1, SIZE, SIZE)
    x_train = kt.tensor(train_x, shape=(args.train,) + shape)
    y_train = kt.tensor(train_y, dtype="int32", shape=(args.train,))
    x_test = kt.tensor(test_x, shape=(args.test,) + shape)
    y_test = kt.tensor(test_y, dtype="int32", shape=(args.test,))
    kt.synchronize()
    t2 = time.time()
    mb = (len(train_x) + len(test_x)) * 4 / 2**20
    print(f"data: {args.train} train / {args.test} test images, generated in {t1 - t0:.1f}s, "
          f"uploaded {mb:.1f} MB in {t2 - t1:.1f}s")

    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 8 x 10 x 10
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16 x 5 x 5
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 64), nn.ReLU(),
        nn.Linear(64, len(CLASSES)),
    )
    print(model)
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    train_loader = DataLoader(x_train, y_train, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(x_test, y_test, batch_size=args.batch)

    print(f"\n{'epoch':>5} {'loss':>8} {'train acc':>10} {'test acc':>9} {'time':>7} {'img/s':>7}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = kt.zeros(1)
        correct = kt.zeros(1, dtype="int32")
        start = time.time()
        for x, y in train_loader:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.detach()
            F.count_correct(logits.detach(), y, correct)
        mean_loss = total_loss.item() / len(train_loader)  # first sync of the epoch
        elapsed = time.time() - start
        train_acc = correct.item() / (len(train_loader) * args.batch)
        test_acc = evaluate(model.eval(), test_loader)
        images = len(train_loader) * args.batch
        print(f"{epoch:>5} {mean_loss:>8.4f} {train_acc:>9.1%} {test_acc:>9.1%} "
              f"{elapsed:>6.1f}s {images / elapsed:>7.0f}")

    # A few test images with predictions
    with kt.no_grad():
        preds = model(x_test[0:4]).tolist()
    print()
    pics = [ascii_image(test_x[i * SIZE * SIZE:(i + 1) * SIZE * SIZE]) for i in range(4)]
    for row in range(SIZE):
        print("   ".join(p[row] for p in pics))
    for i, p in enumerate(preds):
        guess = max(range(len(p)), key=p.__getitem__)
        mark = "ok" if guess == test_y[i] else f"wrong, is {CLASSES[test_y[i]]}"
        print(f"  image {i + 1}: predicted {CLASSES[guess]} ({mark})")


if __name__ == "__main__":
    main()
