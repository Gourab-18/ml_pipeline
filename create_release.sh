#!/bin/bash
# Script to create release v1.0.0

set -e

VERSION="v1.0.0"
RELEASE_DIR="releases/${VERSION}"

echo "🚀 Creating release ${VERSION}..."

# Create release directory
mkdir -p "${RELEASE_DIR}"

# Copy model artifacts (if they exist)
if [ -d "artifacts/models/champion" ]; then
    echo "📦 Copying model artifacts..."
    cp -r artifacts/models/champion "${RELEASE_DIR}/model"
else
    echo "⚠️  No model artifacts found. Run 'make train-full && make export' first."
fi

# Copy configuration
echo "📋 Copying configuration..."
mkdir -p "${RELEASE_DIR}/configs"
cp -r configs/* "${RELEASE_DIR}/configs/" 2>/dev/null || true

# Copy essential source code
echo "💻 Copying source code..."
mkdir -p "${RELEASE_DIR}/src"
cp -r src/* "${RELEASE_DIR}/src/"

# Copy documentation
echo "📚 Copying documentation..."
mkdir -p "${RELEASE_DIR}/docs"
cp -r docs/* "${RELEASE_DIR}/docs/" 2>/dev/null || true
cp README.md "${RELEASE_DIR}/"
cp CONTRIBUTING.md "${RELEASE_DIR}/" 2>/dev/null || true
cp RELEASE.md "${RELEASE_DIR}/" 2>/dev/null || true

# Copy requirements
cp requirements.txt "${RELEASE_DIR}/" 2>/dev/null || true

# Create README for release
cat > "${RELEASE_DIR}/README_RELEASE.md" << EOR
# Release ${VERSION}

This is release ${VERSION} of the Tabular ML Pipeline.

## Contents

- \`model/\` - Champion model artifacts
- \`configs/\` - Configuration files
- \`src/\` - Source code
- \`docs/\` - Documentation
- \`requirements.txt\` - Python dependencies

## Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Quick Start

\`\`\`bash
# Start server
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

# Load model (adjust path as needed)
curl -X POST "http://localhost:8000/load?model_dir=\$(pwd)/model/champion&version=1"
\`\`\`

See README.md for full documentation.
EOR

# Create ZIP archive
echo "📦 Creating ZIP archive..."
cd releases
zip -r "${VERSION}.zip" "${VERSION}/"
cd ..

echo "✅ Release created: releases/${VERSION}/"
echo "✅ ZIP archive: releases/${VERSION}.zip"

# Instructions for git tag
echo ""
echo "📋 Next steps:"
echo "1. Review release artifacts: releases/${VERSION}/"
echo "2. Create git tag: git tag -a ${VERSION} -m \"Release ${VERSION}\""
echo "3. Push tag: git push origin ${VERSION}"
echo "4. Create GitHub release and upload: releases/${VERSION}.zip"

