# Conv2D - 2D Convolution Algorithm Implementation

2D convolution algorithm implemented using NumPy, demonstrating the core operation in convolutional neural networks.

## Quick Start

Use `uv` to run the script directly without manually installing dependencies:

```bash
# Run convolution algorithm demonstration (automatically installs numpy dependency)
uv run conv2d.py
```

The script includes inline dependency declarations (PEP 723), and `uv` will automatically handle dependency installation.

## Other Ways to Run

### Using uv to create virtual environment

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
uv pip install numpy

# Run script
python conv2d.py
```

### Using Python directly

```bash
# Need to install numpy first
pip install numpy
python conv2d.py
```

## File Description

- `conv2d.py` - Efficient implementation using NumPy, supports single and multi-channel convolution
- `pyproject.toml` - Project configuration file

## Features

- 2D convolution operation implementation
- Support for custom stride and padding
- Multiple kernel examples (edge detection, blur, Sobel operator, etc.)
- Fixed input and kernels for demonstration
