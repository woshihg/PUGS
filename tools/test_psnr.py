import cv2
import numpy as np
import math
import argparse

def calculate_psnr(img1, img2):
    """计算两张图片之间的PSNR值。"""
    # 确保图片具有相同的尺寸
    if img1.shape != img2.shape:
        raise ValueError("输入图片的尺寸必须相同。")

    # 将图片转换为浮点数类型以便计算
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 计算均方误差 (MSE)
    mse = np.mean((img1 - img2) ** 2)

    # 如果MSE为0，表示图片完全相同，PSNR为无穷大
    if mse == 0:
        return float('inf')

    # 计算PSNR
    max_pixel_value = 255.0
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))

    return psnr

if __name__ == "__main__":
    gt_image_path = "/home/woshihg/PycharmProjects/PUGS/data-test/mydata/images_aligned/images/test/0006.png"
    pred_image_path = "/home/woshihg/PycharmProjects/PUGS/output/model_2025-12-12_15-41-07/fix/iter_1000/render/0006.png"
    # 读取图片
    gt_img = cv2.imread(gt_image_path)
    pred_img = cv2.imread(pred_image_path)

    if gt_img is None:
        print(f"错误：无法读取基准真值图片: {gt_image_path}")
        exit()

    if pred_img is None:
        print(f"错误：无法读取预测图片: {pred_image_path}")
        exit()

    # 计算并打印PSNR值
    try:
        psnr_value = calculate_psnr(gt_img, pred_img)
        print(f"PSNR: {psnr_value:.4f} dB")
    except ValueError as e:
        print(f"错误: {e}")

# 使用方法:
# python tools/test_psnr.py /path/to/your/gt_image.png /path/to/your/pred_image.png

