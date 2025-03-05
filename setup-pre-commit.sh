#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up pre-commit hooks for this repository...${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed or not in PATH.${NC}"
    echo -e "${YELLOW}Please install Python 3 to continue.${NC}"
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo -e "${RED}pip is not installed or not in PATH.${NC}"
    echo -e "${YELLOW}Please install pip to continue.${NC}"
    exit 1
fi

# Install pre-commit if not already installed
echo -e "${YELLOW}Installing pre-commit if needed...${NC}"
PIP_CMD="pip"
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
fi
$PIP_CMD install pre-commit

# Install the hooks
echo -e "${YELLOW}Installing git hooks...${NC}"
pre-commit install

# Run pre-commit on all files to see if everything is set up correctly
echo -e "${YELLOW}Running hooks on all files to verify setup...${NC}"
pre-commit run --all-files || true

echo -e "${GREEN}Pre-commit hooks setup complete!${NC}"
echo -e "${BLUE}Hooks will now run automatically on git commit.${NC}"
echo -e "${BLUE}You can also run them manually with:${NC}"
echo -e "${YELLOW}    pre-commit run --all-files${NC}"

# Check Node.js (needed for JavaScript hooks)
if ! command -v node &> /dev/null; then
    echo -e "\n${YELLOW}Warning: Node.js not detected.${NC}"
    echo -e "${YELLOW}Some JavaScript hooks may not work without Node.js and npm.${NC}"
    echo -e "${YELLOW}Please install Node.js if you plan to work with JavaScript files.${NC}"
fi

exit 0
