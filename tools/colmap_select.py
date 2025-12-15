import os
import struct
import collections
import numpy as np

# --- 1. 定义 COLMAP 的数据结构 ---
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            xys = np.empty((num_points2D, 2), dtype=np.float64)
            point3D_ids = np.empty((num_points2D,), dtype=np.int64)
            if num_points2D > 0:
                xys_and_point3D_ids = read_next_bytes(fid, num_points2D * 24, "ddq" * num_points2D)
                xys = np.array(xys_and_point3D_ids[0::3] + xys_and_point3D_ids[1::3]).reshape((num_points2D, 2))
                point3D_ids = np.array(xys_and_point3D_ids[2::3])
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def write_images_binary(images, path_to_model_file):
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, *img.qvec.tolist(), "dddd")
            write_next_bytes(fid, *img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            fid.write(img.name.encode("utf-8") + b"\x00")
            write_next_bytes(fid, len(img.xys), "Q")
            for i in range(len(img.xys)):
                write_next_bytes(fid, *img.xys[i].tolist(), "dd")
                write_next_bytes(fid, img.point3D_ids[i], "q")


def write_next_bytes(fid, *args):
    format_str = args[-1]
    data = args[:-1]
    fid.write(struct.pack("<" + format_str, *data))


# --- 2. 核心处理逻辑 ---

def filter_colmap_images(bin_path, keep_image_paths, output_path):
    print(f"正在读取 COLMAP 文件: {bin_path} ...")
    if not os.path.exists(bin_path):
        print(f"错误: 找不到文件 {bin_path}")
        return

    # 读取原始数据
    images = read_images_binary(bin_path)
    print(f"原始包含图像数量: {len(images)}")

    # 提取保留列表的文件名
    keep_names = set(os.path.basename(p) for p in keep_image_paths)
    print(f"参考文件夹中提供的图片数量: {len(keep_names)}")

    # 过滤
    filtered_images = {}
    for image_id, img in images.items():
        # os.path.basename 确保只比对文件名
        bin_img_name = os.path.basename(img.name)
        if bin_img_name in keep_names:
            filtered_images[image_id] = img
        else:
            # 可以取消下面的注释来调试，看看哪些被剔除了
            # print(f"剔除: {bin_img_name}")
            pass

    print(f"过滤后剩余图像数量: {len(filtered_images)}")

    if len(filtered_images) == 0:
        print("警告: 过滤后没有剩余任何图像！请检查 bin 文件里的 name 和图片文件夹里的文件名是否一致。")
        # 如果结果为空，不要覆盖文件，直接返回
        return

    # 写入新文件
    print(f"正在写入新文件: {output_path} ...")
    write_images_binary(filtered_images, output_path)
    print("完成！")


# --- 3. 用户配置区域 ---

if __name__ == "__main__":
    # 1. 原始 bin 路径
    input_bin_path = r"/home/woshihg/PycharmProjects/PUGS/data-test/gsnet/scenes_real/sparse/0/images.bin"

    # 2. 图片文件夹路径
    image_folder_path = r"/home/woshihg/PycharmProjects/PUGS/data-test/gsnet/scenes_real/images"

    # [修正] 获取文件夹下的所有文件名，生成列表
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')  # 支持的后缀
    image_paths_to_keep = [
        f for f in os.listdir(image_folder_path)
        if f.endswith(valid_extensions)
    ]

    # 3. 输出路径
    output_bin_path = r"/home/woshihg/PycharmProjects/PUGS/data-test/gsnet/scenes_real/sparse/0/images.bin"

    print(f"准备开始。检测到参考文件夹有 {len(image_paths_to_keep)} 张图片。")

    filter_colmap_images(input_bin_path, image_paths_to_keep, output_bin_path)