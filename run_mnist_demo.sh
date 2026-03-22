#!/bin/bash
# MNIST Training Demo — Apple Neural Engine (SME2)
#
# This script downloads the MNIST dataset, compiles the demo, and runs it.
# Requirements: Apple Silicon M4 or later, Xcode command line tools
#
# Usage: ./run_mnist_demo.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo ""
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║  MNIST Training Demo — Apple Neural Engine       ║"
echo "  ║  INT8 Quantized • SME2 Bytecode Interpreter      ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo ""
# Download MNIST if needed
if [ ! -f "data/mnist/train-images-idx3-ubyte" ]; then
    echo "Downloading MNIST dataset..."
    python3 scripts/download_mnist.py
    echo ""
fi
# Compile
echo "Compiling..."
mkdir -p build
clang++ -std=c++20 -O2 -march=native \
    -I include \
    -o build/test_mnist \
    tests/test_mnist.cpp \
    2>&1
echo "Done."
echo ""
# Run
time ./build/test_mnist data/mnist
