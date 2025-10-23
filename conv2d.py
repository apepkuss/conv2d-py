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
    Implement 2D convolution operation

    Parameters:
        input_image: Input image (H, W) or (C, H, W)
        kernel: Convolution kernel (K_H, K_W) or (C, K_H, K_W)
        stride: Stride
        padding: Padding

    Returns:
        Output feature map
    """
    # Ensure input is numpy array
    input_image = np.array(input_image)
    kernel = np.array(kernel)

    # Handle single channel case
    if input_image.ndim == 2:
        input_image = input_image[np.newaxis, :, :]
    if kernel.ndim == 2:
        kernel = kernel[np.newaxis, :, :]

    channels, input_height, input_width = input_image.shape
    _, kernel_height, kernel_width = kernel.shape

    # Add padding
    if padding > 0:
        input_padded = np.pad(
            input_image,
            ((0, 0), (padding, padding), (padding, padding)),
            mode='constant',
            constant_values=0
        )
    else:
        input_padded = input_image

    # Calculate output size
    output_height = (input_padded.shape[1] - kernel_height) // stride + 1
    output_width = (input_padded.shape[2] - kernel_width) // stride + 1

    # Initialize output
    output = np.zeros((output_height, output_width))

    # Perform convolution operation
    for i in range(output_height):
        for j in range(output_width):
            # Calculate starting coordinates for current position
            h_start = i * stride
            w_start = j * stride

            # Extract current window
            window = input_padded[:, h_start:h_start+kernel_height, w_start:w_start+kernel_width]

            # Perform element-wise multiplication and sum (all channels)
            output[i, j] = np.sum(window * kernel)

    return output


def main():
    """
    Main function: Test with fixed input and kernel
    """
    print("=" * 60)
    print("2D Convolution Algorithm Demonstration")
    print("=" * 60)

    # 定义固定的输入图像 (5x5)
    input_image = np.array([
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3]
    ], dtype=np.float32)

    print("\nInput image (5x5):")
    print(input_image)

    # Define fixed convolution kernel (3x3) - edge detection kernel
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    print("\nConvolution kernel (3x3) - Edge detection:")
    print(kernel)

    # Test 1: No padding, stride=1
    print("\n" + "=" * 60)
    print("Test 1: stride=1, padding=0")
    print("=" * 60)
    output1 = conv2d(input_image, kernel, stride=1, padding=0)
    print(f"\nOutput feature map ({output1.shape[0]}x{output1.shape[1]}):")
    print(output1)

    # Test 2: With padding, stride=1
    print("\n" + "=" * 60)
    print("Test 2: stride=1, padding=1")
    print("=" * 60)
    output2 = conv2d(input_image, kernel, stride=1, padding=1)
    print(f"\nOutput feature map ({output2.shape[0]}x{output2.shape[1]}):")
    print(output2)

    # Test 3: No padding, stride=2
    print("\n" + "=" * 60)
    print("Test 3: stride=2, padding=0")
    print("=" * 60)
    output3 = conv2d(input_image, kernel, stride=2, padding=0)
    print(f"\nOutput feature map ({output3.shape[0]}x{output3.shape[1]}):")
    print(output3)

    # Test 4: Using different kernel - average blur kernel
    print("\n" + "=" * 60)
    print("Test 4: Average blur kernel (3x3)")
    print("=" * 60)
    kernel_blur = np.ones((3, 3), dtype=np.float32) / 9.0
    print("\nConvolution kernel (average blur):")
    print(kernel_blur)
    output4 = conv2d(input_image, kernel_blur, stride=1, padding=0)
    print(f"\nOutput feature map ({output4.shape[0]}x{output4.shape[1]}):")
    print(output4)

    # Test 5: Multi-channel input
    print("\n" + "=" * 60)
    print("Test 5: Multi-channel input (RGB image)")
    print("=" * 60)
    # Create a 3-channel 4x4 image
    input_rgb = np.array(
        [
            # R channel
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            # G channel
            [[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]],
            # B channel
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ],
        dtype=np.float32,
    )

    print(f"\nInput RGB image (3x4x4):")
    print("R channel:\n", input_rgb[0])
    print("G channel:\n", input_rgb[1])
    print("B channel:\n", input_rgb[2])

    # Multi-channel convolution kernel
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

    print("\nConvolution kernel (vertical edge detection, 3 channels):")
    print(kernel_rgb[0])

    output5 = conv2d(input_rgb, kernel_rgb, stride=1, padding=0)
    print(f"\nOutput feature map ({output5.shape[0]}x{output5.shape[1]}):")
    print(output5)

    print("\n" + "=" * 60)
    print("Demonstration completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
