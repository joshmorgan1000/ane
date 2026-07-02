#!/bin/bash
# Apple Silicon SME/SME2 Hardware Dashboard Build Script
#
# Author: Josh Morgan (@joshmorgan1000 on GitHub) with help from Claude and Gemini
# Released under the MIT License
set -e
# Ensure we're in the repository root
cd "$(dirname "$0")"
NODE_REQUIREMENT="Node.js 20.19+ or 22.12+"
NODE_INSTALL_HINT="Install from https://nodejs.org or run: brew install node"
CMAKE_REQUIREMENT="CMake 3.19+"
PYTHON_REQUIREMENT="Python 3.10+"
PYTHON_VENV_DIR="$(pwd)/.dashboard-venv"
print_missing_prerequisite() {
    echo -e "\033[1;31mMissing: $1\033[0m"
    echo "  $2"
}
print_prerequisite_footer() {
    echo " "
    echo "Install the missing tools above, then run ./dashboard.sh again."
    echo "Homebrew is optional but convenient for developer tools: https://brew.sh"
}
check_cmake_version() {
    local cmake_line
    local cmake_version
    local cmake_major
    local cmake_minor
    cmake_line=$(cmake --version 2>/dev/null)
    cmake_line="${cmake_line%%$'\n'*}"
    cmake_version="${cmake_line#cmake version }"
    cmake_major="${cmake_version%%.*}"
    cmake_minor="${cmake_version#*.}"
    cmake_minor="${cmake_minor%%.*}"
    if [[ ! "$cmake_major" =~ ^[0-9]+$ || ! "$cmake_minor" =~ ^[0-9]+$ ]]; then
        print_missing_prerequisite "$CMAKE_REQUIREMENT" "Install with Homebrew: brew install cmake"
        return 1
    fi
    if (( cmake_major > 3 )); then
        return 0
    fi
    if (( cmake_major == 3 && cmake_minor >= 19 )); then
        return 0
    fi
    print_missing_prerequisite "$CMAKE_REQUIREMENT found ${cmake_version}" "Install with Homebrew: brew install cmake"
    return 1
}
check_node_version() {
    local node_version
    local node_major
    local node_minor
    node_version=$(node -v 2>/dev/null)
    node_version="${node_version#v}"
    node_major="${node_version%%.*}"
    node_minor="${node_version#*.}"
    node_minor="${node_minor%%.*}"
    if [[ ! "$node_major" =~ ^[0-9]+$ ]]; then
        print_missing_prerequisite "$NODE_REQUIREMENT" "$NODE_INSTALL_HINT"
        return 1
    fi
    if [[ ! "$node_minor" =~ ^[0-9]+$ ]]; then
        node_minor=0
    fi
    if (( node_major == 20 && node_minor >= 19 )); then
        return 0
    fi
    if (( node_major == 22 && node_minor >= 12 )); then
        return 0
    fi
    if (( node_major > 22 )); then
        return 0
    fi
    print_missing_prerequisite "$NODE_REQUIREMENT found v${node_version}" "$NODE_INSTALL_HINT"
    return 1
}
check_python_version() {
    local python_line
    local python_version
    local python_major
    local python_minor
    python_line=$(python3 -V 2>&1)
    python_version="${python_line#Python }"
    python_major="${python_version%%.*}"
    python_minor="${python_version#*.}"
    python_minor="${python_minor%%.*}"
    if [[ ! "$python_major" =~ ^[0-9]+$ || ! "$python_minor" =~ ^[0-9]+$ ]]; then
        print_missing_prerequisite "$PYTHON_REQUIREMENT" "Install from https://www.python.org/downloads/macos/ or run: brew install python"
        return 1
    fi
    if (( python_major > 3 )); then
        return 0
    fi
    if (( python_major == 3 && python_minor >= 10 )); then
        return 0
    fi
    print_missing_prerequisite "$PYTHON_REQUIREMENT found ${python_version}" "Install from https://www.python.org/downloads/macos/ or run: brew install python"
    return 1
}
check_python_venv_support() {
    if ! python3 -m venv --help >/dev/null 2>&1; then
        print_missing_prerequisite "Python venv support" "Install a Python build with venv support: brew install python"
        return 1
    fi
    return 0
}
ensure_python_environment() {
    local venv_python
    local created_venv=0
    echo "🐍 Preparing dashboard Python environment..."
    if [[ ! -x "${PYTHON_VENV_DIR}/bin/python3" ]]; then
        python3 -m venv "$PYTHON_VENV_DIR"
        created_venv=1
    fi
    venv_python="${PYTHON_VENV_DIR}/bin/python3"
    if (( created_venv != 0 )); then
        "$venv_python" -m pip install --upgrade pip
    fi
    if ! "$venv_python" -c "import torch; import torchvision" >/dev/null 2>&1; then
        echo "📦 Installing PyTorch dashboard packages..."
        if ! "$venv_python" -m pip install torch torchvision; then
            echo -e "\033[1;31mFailed to install PyTorch dashboard packages.\033[0m"
            echo "Check your network connection, then run ./dashboard.sh again."
            exit 1
        fi
    fi
    export VIRTUAL_ENV="$PYTHON_VENV_DIR"
    export PATH="${PYTHON_VENV_DIR}/bin:$PATH"
    echo -e "\033[0;32mDashboard Python environment ready.\033[0m"
}
check_dashboard_prerequisites() {
    local missing=0
    echo "Checking dashboard build prerequisites..."
    if [[ "$OSTYPE" != "darwin"* ]]; then
        print_missing_prerequisite "macOS" "This dashboard is intended for Apple Silicon machines with SME/SME2."
        missing=1
    fi
    if ! xcode-select -p >/dev/null 2>&1; then
        print_missing_prerequisite "Xcode Command Line Tools" "Install with: xcode-select --install"
        missing=1
    fi
    if ! xcrun --find clang >/dev/null 2>&1; then
        print_missing_prerequisite "Apple clang" "Install or repair Xcode Command Line Tools with: xcode-select --install"
        missing=1
    fi
    if ! xcrun --find clang++ >/dev/null 2>&1; then
        print_missing_prerequisite "Apple clang++" "Install or repair Xcode Command Line Tools with: xcode-select --install"
        missing=1
    fi
    if ! command -v cmake >/dev/null 2>&1; then
        print_missing_prerequisite "$CMAKE_REQUIREMENT" "Install with Homebrew: brew install cmake"
        missing=1
    else
        if ! check_cmake_version; then
            missing=1
        fi
    fi
    if ! command -v ninja >/dev/null 2>&1 && ! command -v make >/dev/null 2>&1; then
        print_missing_prerequisite "ninja or make" "Install Ninja with: brew install ninja, or install Xcode Command Line Tools for make."
        missing=1
    fi
    if ! command -v node >/dev/null 2>&1; then
        print_missing_prerequisite "$NODE_REQUIREMENT" "$NODE_INSTALL_HINT"
        missing=1
    else
        if ! check_node_version; then
            missing=1
        fi
    fi
    if ! command -v npm >/dev/null 2>&1; then
        print_missing_prerequisite "npm" "$NODE_INSTALL_HINT"
        missing=1
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        print_missing_prerequisite "$PYTHON_REQUIREMENT" "Install from https://www.python.org/downloads/macos/ or run: brew install python"
        missing=1
    else
        if ! check_python_version; then
            missing=1
        fi
        if ! check_python_venv_support; then
            missing=1
        fi
    fi
    if (( missing != 0 )); then
        print_prerequisite_footer
        exit 1
    fi
    echo -e "\033[0;32mAll required dashboard build tools found.\033[0m"
}
check_dashboard_prerequisites
# Detect Chip Type
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -o 'M[0-9]' | head -n 1)
if [ -z "$CHIP" ]; then
    echo -e "\033[1;33mWarning: Could not detect Apple Silicon chip type. Using 'unknown' as chip name.\033[0m"
    CHIP="unknown"
fi
echo -e " "
echo -e "\033[35m────────\033[36m Apple Silicon $CHIP Hardware Dashboard \033[35m────────\033[0m"
echo " "
echo "This script will take ~15-20 minutes to run since it performs live hardware probes,"
echo "cools down between contention tests, and builds a standalone React dashboard package."
echo "Are you sure you want to continue? (y/n)"
read -n 1 -s answer
echo " "
if [[ "$answer" != "y" ]]; then
    echo "Aborting dashboard build."
    exit 0
fi
ensure_python_environment
cd sme-ui
# Ensure dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing UI dependencies..."
    npm install
fi
# Run the MNIST comparison before the long probe and throughput pass
echo "🏋️  Running MNIST SME/PyTorch comparison..."
npm run fetch-mnist
# Run the backend probes to grab live CPU limits
echo "🔍 Running hardware probes..."
npm run fetch-probe
# Run contention tests after short probes and comparisons
echo "🔥 Running throughput contention tests..."
npm run fetch-throughput
# Build the react app down to a single HTML package
echo "🏗️  Building standalone UI package..."
npm run build
echo "✅ Done! Publishing dashboard..."
mkdir -p ../dashboards
OUT_FILE="../dashboards/local_${CHIP}_results.html"
cp dist/index.html "$OUT_FILE"
echo "Saved locally to: $OUT_FILE"
# Open the static packaged HTML file.
echo -e " "
echo -e "\033[35m────────\033[36m Done! \033[35m────────\033[0m"
echo " "
echo "If your browser doesn't open automatically, you can find the dashboard at: $OUT_FILE"
echo -e "\033[36mCheers! 🥂\033[0m"
echo ""
open "$OUT_FILE"
