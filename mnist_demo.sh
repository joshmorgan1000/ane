#!/bin/bash
# MNIST Training Demo — Apple Neural Engine (SME2)
#
# This script downloads the MNIST dataset, builds the demo, and runs it.
# Requirements: Apple Silicon M4 or later, Xcode command line tools, CMake, Ninja
#
# Usage: ./mnist_demo.sh [--clean] [--build-only]
#   --clean       Wipe the build directory and reconfigure from scratch
#   --build-only  Build but do not run the demo
#
# Author: Josh Morgan (@joshmorgan1000 on GitHub) with help from Claude and Gemini
# Released under the MIT License
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CLEAN=0
BUILD_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --clean)      CLEAN=1 ;;
        --build-only) BUILD_ONLY=1 ;;
    esac
done
echo ""
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║  MNIST Training Demo — Apple Neural Engine       ║"
echo "  ║  FP32 FMOPA • SME2 Bytecode Interpreter          ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo ""
# Clean if requested
if [ "$CLEAN" -eq 1 ]; then
    echo "Cleaning build directory..."
    rm -rf build
    echo ""
fi
# Download MNIST if needed
if [ ! -f "data/mnist/train-images-idx3-ubyte" ]; then
    echo "Downloading MNIST dataset..."
    python3 scripts/download_mnist.py
    echo ""
fi
# Configure if needed
if [ ! -f "build/build.ninja" ]; then
    echo "Configuring build..."
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
fi
# Build
echo "Building..."
cmake --build build --target test_mnist
echo "Done."
echo ""
if [ "$BUILD_ONLY" -eq 1 ]; then
    exit 0
fi
# Run
time ./build/bin/test_mnist data/mnist
