#!/usr/bin/env python3
"""Download and extract MNIST dataset to data/mnist/ directory.

Downloads from the original Yann LeCun mirror and extracts the raw
binary files. No external dependencies required — uses only stdlib.
"""
import os
import gzip
import struct
import urllib.request
import sys

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte.gz":  "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte.gz":  "t10k-labels-idx1-ubyte",
}

def download_and_extract(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for gz_name, raw_name in FILES.items():
        raw_path = os.path.join(out_dir, raw_name)
        if os.path.exists(raw_path):
            print(f"  {raw_name} already exists, skipping")
            continue
        url = MIRROR + gz_name
        gz_path = os.path.join(out_dir, gz_name)
        print(f"  Downloading {gz_name}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, gz_path)
        print("done")
        print(f"  Extracting {raw_name}...", end=" ", flush=True)
        with gzip.open(gz_path, "rb") as f_in:
            with open(raw_path, "wb") as f_out:
                f_out.write(f_in.read())
        os.remove(gz_path)
        print("done")
    # Verify file integrity
    with open(os.path.join(out_dir, "train-images-idx3-ubyte"), "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Bad magic: {magic}"
        assert n == 60000, f"Expected 60000 images, got {n}"
        assert rows == 28 and cols == 28
    with open(os.path.join(out_dir, "train-labels-idx1-ubyte"), "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Bad magic: {magic}"
        assert n == 60000
    with open(os.path.join(out_dir, "t10k-images-idx3-ubyte"), "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        assert n == 10000
    with open(os.path.join(out_dir, "t10k-labels-idx1-ubyte"), "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        assert n == 10000
    print(f"  Verified: 60000 train + 10000 test images (28x28)")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    out_dir = os.path.join(project_root, "data", "mnist")
    print("MNIST Dataset Download")
    download_and_extract(out_dir)
    print("Done!")
