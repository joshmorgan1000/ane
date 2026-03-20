#!/bin/bash

# Ensure we're in the repository root
cd "$(dirname "$0")"

echo "==============================================="
echo "   Apple M4 SME/SME2 Hardare Dashboard Sync    "
echo "==============================================="

cd m4-sme-docs

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

echo "✅ Done! Opening dashboard..."
# Open the static packaged HTML file.
# macOS default:
open dist/index.html
