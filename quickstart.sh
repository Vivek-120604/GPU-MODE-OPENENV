#!/bin/bash
# Quick Start Guide for BiologicalOptimizationEnv

echo "=========================================="
echo "BiologicalOptimizationEnv - Quick Start"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python
echo "${BLUE}[1] Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.11 or higher."
    exit 1
fi
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "${GREEN}✓ Python ${python_version}${NC}"
echo ""

# Setup virtual environment
echo "${BLUE}[2] Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "${GREEN}✓ Virtual environment created${NC}"
else
    echo "${GREEN}✓ Virtual environment already exists${NC}"
fi
source venv/bin/activate
echo ""

# Install dependencies
echo "${BLUE}[3] Installing dependencies...${NC}"
pip install -q fastapi uvicorn pydantic numpy openai requests python-dotenv pyyaml
echo "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Run validation
echo "${BLUE}[4] Running validation tests...${NC}"
python validate.py
echo ""

# Show next steps
echo "${BLUE}[5] Next steps:${NC}"
echo ""
echo "  ${GREEN}To start the server:${NC}"
echo "    python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload"
echo ""
echo "  ${GREEN}To test the API (in another terminal):${NC}"
echo "    curl http://localhost:7860/"
echo "    curl -X POST http://localhost:7860/reset -H 'Content-Type: application/json' -d '{\"task\": \"easy\", \"seed\": 42}'"
echo ""
echo "  ${GREEN}To run inference (requires HF_TOKEN):${NC}"
echo "    export HF_TOKEN=\"your-huggingface-token\""
echo "    python inference.py --task easy --episodes 1"
echo ""
echo "  ${GREEN}To view API documentation:${NC}"
echo "    Open http://localhost:7860/docs in your browser"
echo ""
echo "${GREEN}✓ Setup complete!${NC}"
