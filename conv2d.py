#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "numpy>=1.20.0",
# ]
# ///

import numpy as np


def conv2d(input_image, kernel, stride=1, padding=0):
    """
    实现2D卷积操作

    参数:
        input_image: 输入图像 (H, W) 或 (C, H, W)
        kernel: 卷积核 (K_H, K_W) 或 (C, K_H, K_W)
        stride: 步长
        padding: 填充

    返回:
        输出特征图
    """
    # 确保输入是numpy数组
    input_image = np.array(input_image)
    kernel = np.array(kernel)

    # 处理单通道情况
    if input_image.ndim == 2:
        input_image = input_image[np.newaxis, :, :]
    if kernel.ndim == 2:
        kernel = kernel[np.newaxis, :, :]

    channels, input_height, input_width = input_image.shape
    _, kernel_height, kernel_width = kernel.shape

    # 添加填充
    if padding > 0:
        input_padded = np.pad(
            input_image,
            ((0, 0), (padding, padding), (padding, padding)),
            mode='constant',
            constant_values=0
        )
    else:
        input_padded = input_image

    # 计算输出尺寸
    output_height = (input_padded.shape[1] - kernel_height) // stride + 1
    output_width = (input_padded.shape[2] - kernel_width) // stride + 1

    # 初始化输出
    output = np.zeros((output_height, output_width))

    # 执行卷积操作
    for i in range(output_height):
        for j in range(output_width):
            # 计算当前位置的起始坐标
            h_start = i * stride
            w_start = j * stride

            # 提取当前窗口
            window = input_padded[:, h_start:h_start+kernel_height, w_start:w_start+kernel_width]

            # 执行逐元素乘法并求和（所有通道）
            output[i, j] = np.sum(window * kernel)

    return output


def main():
    """
    主函数：使用固定的输入和卷积核进行测试
    """
    print("=" * 60)
    print("2D卷积算法演示")
    print("=" * 60)

    # 定义固定的输入图像 (5x5)
    input_image = np.array([
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3]
    ], dtype=np.float32)

    print("\n输入图像 (5x5):")
    print(input_image)

    # 定义固定的卷积核 (3x3) - 边缘检测核
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    print("\n卷积核 (3x3) - 边缘检测:")
    print(kernel)

    # 测试1: 无填充，步长为1
    print("\n" + "=" * 60)
    print("测试1: stride=1, padding=0")
    print("=" * 60)
    output1 = conv2d(input_image, kernel, stride=1, padding=0)
    print(f"\n输出特征图 ({output1.shape[0]}x{output1.shape[1]}):")
    print(output1)

    # 测试2: 有填充，步长为1
    print("\n" + "=" * 60)
    print("测试2: stride=1, padding=1")
    print("=" * 60)
    output2 = conv2d(input_image, kernel, stride=1, padding=1)
    print(f"\n输出特征图 ({output2.shape[0]}x{output2.shape[1]}):")
    print(output2)

    # 测试3: 无填充，步长为2
    print("\n" + "=" * 60)
    print("测试3: stride=2, padding=0")
    print("=" * 60)
    output3 = conv2d(input_image, kernel, stride=2, padding=0)
    print(f"\n输出特征图 ({output3.shape[0]}x{output3.shape[1]}):")
    print(output3)

    # 测试4: 使用不同的卷积核 - 平均模糊核
    print("\n" + "=" * 60)
    print("测试4: 平均模糊核 (3x3)")
    print("=" * 60)
    kernel_blur = np.ones((3, 3), dtype=np.float32) / 9.0
    print("\n卷积核 (平均模糊):")
    print(kernel_blur)
    output4 = conv2d(input_image, kernel_blur, stride=1, padding=0)
    print(f"\n输出特征图 ({output4.shape[0]}x{output4.shape[1]}):")
    print(output4)

    # 测试5: 多通道输入
    print("\n" + "=" * 60)
    print("测试5: 多通道输入 (RGB图像)")
    print("=" * 60)
    # 创建一个3通道的4x4图像
    input_rgb = np.array([
        # R通道
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]],
        # G通道
        [[16, 15, 14, 13],
         [12, 11, 10, 9],
         [8, 7, 6, 5],
         [4, 3, 2, 1]],
        # B通道
        [[1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3],
         [4, 4, 4, 4]]
    ], dtype=np.float32)

    print(f"\n输入RGB图像 (3x4x4):")
    print("R通道:\n", input_rgb[0])
    print("G通道:\n", input_rgb[1])
    print("B通道:\n", input_rgb[2])

    # 多通道卷积核
    kernel_rgb = np.array([
        [[1, 0, -1],
         [1, 0, -1],
         [1, 0, -1]],
        [[1, 0, -1],
         [1, 0, -1],
         [1, 0, -1]],
        [[1, 0, -1],
         [1, 0, -1],
         [1, 0, -1]]
    ], dtype=np.float32) / 3.0

    print("\n卷积核 (垂直边缘检测, 3通道):")
    print(kernel_rgb[0])

    output5 = conv2d(input_rgb, kernel_rgb, stride=1, padding=0)
    print(f"\n输出特征图 ({output5.shape[0]}x{output5.shape[1]}):")
    print(output5)

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
