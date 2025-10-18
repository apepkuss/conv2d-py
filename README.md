# Conv2D - 2D卷积算法实现

使用NumPy实现的2D卷积算法，展示了卷积神经网络中的核心操作。

## 快速开始

使用 `uv` 可以直接运行脚本，无需手动安装依赖：

```bash
# 运行卷积算法演示（自动安装numpy依赖）
uv run conv2d.py
```

脚本包含了内联依赖声明（PEP 723），`uv` 会自动处理依赖安装。

## 其他运行方式

### 使用 uv 创建虚拟环境

```bash
# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 安装依赖
uv pip install numpy

# 运行脚本
python conv2d.py
```

### 直接使用 Python

```bash
# 需要先安装 numpy
pip install numpy
python conv2d.py
```

## 文件说明

- `conv2d.py` - 使用NumPy的高效实现，支持单通道和多通道卷积
- `pyproject.toml` - 项目配置文件

## 功能特性

- 2D卷积操作实现
- 支持自定义步长（stride）和填充（padding）
- 多种卷积核示例（边缘检测、模糊、Sobel算子等）
- 固定输入和卷积核用于演示
