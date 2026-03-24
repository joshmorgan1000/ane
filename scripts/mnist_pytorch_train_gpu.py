#!/usr/bin/env python3
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# -----------------------------
# Data loading (no DataLoader; match JAX)
# -----------------------------

def load_mnist_tensor(root="./tests/mnist", device="mps"):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    # Tensor form: uint8 [0,255]
    x = train_dataset.data.to(torch.float32) / 255.0  # (60000, 28, 28), in [0,1]
    y = train_dataset.targets.to(torch.long)          # (60000,)

    # Flatten to 784
    x = x.view(-1, 784)

    # Move to device once
    x = x.to(device)
    y = y.to(device)
    return x, y


def batch_iterator(x, y, batch_size, shuffle=True):
    n = x.size(0)
    if shuffle:
        perm = torch.randperm(n, device=x.device)
        x = x[perm]
        y = y[perm]
    for i in range(0, n, batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]


# -----------------------------
# Model
# -----------------------------

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x: (batch, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# -----------------------------
# Training loop
# -----------------------------

def train(use_compile: bool = False):
    device = torch.device("cpu")

    batch_size = 50
    num_epochs = 10
    lr = 0.01

    print(f"Loading MNIST to {device}...")
    x_train, y_train = load_mnist_tensor(device=device)

    model = MNISTNet().to(device)
    if use_compile:
        # Requires PyTorch 2.x
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    mode_name = "PyTorch MPS (compile)" if use_compile else "PyTorch MPS (eager)"

    print(f"Starting {mode_name} MNIST Training")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Optimizer: SGD")
    print(f"Architecture: 784 -> 128 (ReLU) -> 10 (Softmax via CrossEntropy)")
    print(f"torch.compile: {use_compile}")
    print("-" * 60)

    model.train()
    total_samples = 0
    start_time = time.time()

    n = x_train.size(0)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for batch_idx, (data, target) in enumerate(batch_iterator(x_train, y_train, batch_size, shuffle=True)):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            bs = data.size(0)
            epoch_loss += loss.item() * bs
            pred = output.argmax(dim=1)
            epoch_correct += pred.eq(target).sum().item()
            epoch_samples += bs
            total_samples += bs

            if (batch_idx + 1) % 150 == 0:
                avg_loss = epoch_loss / epoch_samples
                accuracy = 100.0 * epoch_correct / epoch_samples
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Samples: {total_samples:6d} | "
                    f"Avg Loss: {avg_loss:.5f} | "
                    f"Accuracy: {accuracy:.1f}%"
                )

        avg_loss = epoch_loss / epoch_samples
        accuracy = 100.0 * epoch_correct / epoch_samples
        print(
            f"Epoch {epoch+1}/{num_epochs} Complete | "
            f"Total Samples: {total_samples:6d} | "
            f"Avg Loss: {avg_loss:.5f} | "
            f"Accuracy: {accuracy:.1f}%"
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    samples_per_sec = total_samples / elapsed_time

    print("-" * 60)
    print(f"Training Complete! ({mode_name})")
    print(f"Total samples processed: {total_samples}")
    print(f"Total time: {elapsed_time:.3f} seconds")
    print(f"Throughput: {samples_per_sec:.1f} samples/sec")
    print("-" * 60)


if __name__ == "__main__":
    train(use_compile=False)
    # train(use_compile=True)