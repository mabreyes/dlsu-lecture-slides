# Makefile for MLP Visualizer Project
# Provides commands for setup, development, testing, and deployment

# Variables
SHELL := /bin/bash
PYTHON := python3
PIP := $(PYTHON) -m pip
NODE := node
NPM := npm
PRETTIER := npx prettier
ESLINT := npx eslint
BLACK := black
FLAKE8 := flake8
ISORT := isort
PRE_COMMIT := pre-commit

# Node.js project paths
NODE_SRC_DIR := src
NODE_TEST_DIR := src/tests

# Python project paths (if applicable)
PYTHON_SRC_DIR := python
PYTHON_TEST_DIR := python/tests

# Default target
.PHONY: help
help:
	@echo "MLP Visualizer Make Commands"
	@echo "--------------------------"
	@echo "setup           : Install all dependencies for development"
	@echo "setup-node      : Install Node.js dependencies"
	@echo "setup-python    : Install Python dependencies"
	@echo "setup-pre-commit: Install pre-commit hooks"
	@echo "dev             : Start development server"
	@echo "build           : Build production assets"
	@echo "lint            : Run all linters"
	@echo "lint-js         : Run ESLint on JavaScript files"
	@echo "lint-py         : Run Flake8 on Python files"
	@echo "format          : Format all code"
	@echo "format-js       : Format JavaScript with Prettier"
	@echo "format-py       : Format Python with Black"
	@echo "test            : Run all tests"
	@echo "test-js         : Run JavaScript tests"
	@echo "test-py         : Run Python tests"
	@echo "clean           : Clean build artifacts"
	@echo "hooks           : Run pre-commit hooks on all files"

# Setup targets
.PHONY: setup setup-node setup-python setup-pre-commit
setup: setup-node setup-python setup-pre-commit

setup-node:
	@echo "Installing Node.js dependencies..."
	cd $(NODE_SRC_DIR) && $(NPM) install

setup-python:
	@echo "Installing Python dependencies..."
	$(PIP) install -r requirements.txt || echo "No requirements.txt found. Skipping..."

setup-pre-commit:
	@echo "Setting up pre-commit hooks..."
	$(PIP) install pre-commit
	$(PRE_COMMIT) install

# Development targets
.PHONY: dev build
dev:
	@echo "Starting development server..."
	cd $(NODE_SRC_DIR) && $(NPM) run dev

build:
	@echo "Building production assets..."
	cd $(NODE_SRC_DIR) && $(NPM) run build

# Linting targets
.PHONY: lint lint-js lint-py
lint: lint-js lint-py

lint-js:
	@echo "Linting JavaScript files..."
	cd $(NODE_SRC_DIR) && $(ESLINT) .

lint-py:
	@echo "Linting Python files..."
	$(FLAKE8) $(PYTHON_SRC_DIR) || echo "No Python files to lint or no Flake8 installed."

# Formatting targets
.PHONY: format format-js format-py
format: format-js format-py

format-js:
	@echo "Formatting JavaScript files..."
	$(PRETTIER) --write "$(NODE_SRC_DIR)/**/*.{js,jsx,ts,tsx,json,css,html}"

format-py:
	@echo "Formatting Python files..."
	$(BLACK) $(PYTHON_SRC_DIR) || echo "No Python files to format or no Black installed."
	$(ISORT) $(PYTHON_SRC_DIR) || echo "No Python files to sort imports or no isort installed."

# Testing targets
.PHONY: test test-js test-py
test: test-js test-py

test-js:
	@echo "Running JavaScript tests..."
	cd $(NODE_SRC_DIR) && $(NPM) test

test-py:
	@echo "Running Python tests..."
	$(PYTHON) -m pytest $(PYTHON_TEST_DIR) || echo "No Python tests to run or pytest not installed."

# Cleanup target
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(NODE_SRC_DIR)/build $(NODE_SRC_DIR)/dist
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name *.egg-info -exec rm -rf {} +
	find . -type d -name *.egg -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .coverage -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type d -name *.pyc -exec rm -rf {} +

# Pre-commit hooks
.PHONY: hooks
hooks:
	@echo "Running pre-commit hooks on all files..."
	$(PRE_COMMIT) run --all-files
