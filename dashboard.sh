#!/bin/bash
# Apple Silicon SME/SME2 Hardware Dashboard Build Script
#
# Author: Josh Morgan (@joshmorgan1000 on GitHub) with help from Claude and Gemini
# Released under the MIT License
# Ensure we're in the repository root
cd "$(dirname "$0")"
# Detect Chip Type
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -o 'M[0-9]' | head -n 1)
if [ -z "$CHIP" ]; then
    CHIP="Silicon"
fi
echo -e " "
echo -e "\033[35m______________________________________________________\033[0m"
echo -e "\033[36m     __     _____ ______  _______ __   _ _______\033[0m"
echo -e "\033[36m     |        |   |_____] |_____| | \  | |______\033[0m"
echo -e "\033[36m     |_____ __|__ |_____] |     | |  \_| |______\033[0m"
echo -e "\033[35m──────────────────────────────────────────────────────\033[0m"
echo -e "\033[33m       Apple Silicon $CHIP Hardware Dashboard\033[0m"
echo " "
echo "This script will take ~5 minutes to run since it performs live hardware probes"
echo "and builds a standalone React dashboard package."
echo "Are you sure you want to continue? (y/n)"
read -n 1 -s answer
echo " "
if [[ "$answer" != "y" ]]; then
    echo "Aborting dashboard build."
    exit 0
fi
cd sme-ui
# Ensure dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing UI dependencies..."
    npm install
fi
# Run the backend probe to grab live CPU limits
echo "🔍 Running hardware probes..."
npm run fetch-data
# Build the react app down to a single HTML package
echo "🏗️  Building standalone UI package..."
npm run build
echo "✅ Done! Publishing dashboard..."
mkdir -p ../dashboards
OUT_FILE="../dashboards/local_${CHIP}_results.html"
cp dist/index.html "$OUT_FILE"
echo "Saved locally to: $OUT_FILE"
# Open the static packaged HTML file.
open "$OUT_FILE"
