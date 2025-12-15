import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1, img2):
    """计算两张图片之间的SSIM值。"""
    # 确保图片具有相同的尺寸
    if img1.shape != img2.shape:
        raise ValueError("输入图片的尺寸必须相同。")

    # 为了计算SSIM，最好将图片转为灰度图，因为SSIM主要关注结构信息
    # 如果要对彩色图片计算，需要设置multichannel=True
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # ssim_value = ssim(gray1, gray2)

    # 对多通道（彩色）图片计算SSIM
    # scikit-image < 0.19: use `multichannel=True`
    # scikit-image >= 0.19: use `channel_axis=-1`
    (ssim_value, _) = ssim(img1, img2, full=True, channel_axis=-1)


    return ssim_value

if __name__ == "__main__":
    gt_image_path = "/home/woshihg/PycharmProjects/PUGS/data-test/mydata/images_aligned/images/test/0006.png"
    # pred_image_path = "/home/woshihg/PycharmProjects/PUGS/output/model_2025-12-12_15-41-07/fix/iter_1000/render/0006.png"
    pred_image_path = "/home/woshihg/PycharmProjects/PUGS/output/model_2025-12-10_13-22-28/point_cloud/iteration_1000/20251212-170624.jpg"

    # 读取图片
    gt_img = cv2.imread(gt_image_path)
    pred_img = cv2.imread(pred_image_path)

    if gt_img is None:
        print(f"错误：无法读取基准真值图片: {gt_image_path}")
        exit()

    if pred_img is None:
        print(f"错误：无法读取预测图片: {pred_image_path}")
        exit()

    # 计算并打印SSIM值
    try:
        ssim_val = calculate_ssim(gt_img, pred_img)
        print(f"SSIM: {ssim_val:.4f}")
    except ValueError as e:
        print(f"错误: {e}")
    except ImportError:
        print("错误: 请安装 scikit-image 库 (pip install scikit-image)")

# 使用方法:
# python tools/test_ssim.py

