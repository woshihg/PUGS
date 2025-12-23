#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.enhance_utils import filter_random_init_points, get_initial_sacle
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: any
    features: torch.tensor
    masks: torch.tensor
    mask_scales: torch.tensor
    image_path: str
    image_name: str
    width: int
    height: int
    cx: float = None
    cy: float = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, features_folder = None, masks_folder = None, mask_scale_folder = None, sample_rate = 1.0, allow_principle_point_shift = False):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        if idx % 10 >= sample_rate * 10:
            continue

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        if not os.path.exists(image_path):
            continue

        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(f"Reading camera {idx+1}/{len(cam_extrinsics)} from {images_folder}")
        sys.stdout.flush()

        # ================== 修改 1: 提前读取图片以获取实际分辨率 ==================
        image_name = os.path.basename(image_path).split(".")[0]

        # 使用 PIL 读取实际图片
        image = Image.open(image_path)
        actual_width, actual_height = image.size  # 获取当前图片的真实尺寸

        # 获取 COLMAP 记录的原始尺寸
        colmap_width = intr.width
        colmap_height = intr.height

        # 计算缩放比例 (Scale Factor)
        scale_x = actual_width / colmap_width
        scale_y = actual_height / colmap_height

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # ================== 修改 2: 根据缩放比例调整内参 ==================
        # 初始化 cx, cy (主点)
        cx = colmap_width / 2.0
        cy = colmap_height / 2.0

        if intr.model == "SIMPLE_PINHOLE":
            # params: [f, cx, cy]
            raw_f = intr.params[0]
            raw_cx = intr.params[1]
            raw_cy = intr.params[2]

            # 缩放
            focal_length_x = raw_f * scale_x
            focal_length_y = raw_f * scale_y  # SIMPLE_PINHOLE 只有一个f，通常假设方形像素
            cx = raw_cx * scale_x
            cy = raw_cy * scale_y

        elif intr.model == "PINHOLE":
            # params: [fx, fy, cx, cy]
            raw_fx = intr.params[0]
            raw_fy = intr.params[1]
            raw_cx = intr.params[2]
            raw_cy = intr.params[3]

            # 缩放
            focal_length_x = raw_fx * scale_x
            focal_length_y = raw_fy * scale_y
            cx = raw_cx * scale_x
            cy = raw_cy * scale_y

        elif intr.model == "SIMPLE_RADIAL":
            # params: [f, cx, cy, k]
            raw_f = intr.params[0]
            raw_cx = intr.params[1]
            raw_cy = intr.params[2]

            # 缩放 (SIMPLE_RADIAL 也是单焦距)
            focal_length_x = raw_f * scale_x
            focal_length_y = raw_f * scale_y
            cx = raw_cx * scale_x
            cy = raw_cy * scale_y

        else:
            assert False, f"Colmap camera model {intr.model} not handled!"

        # ================== 修改 3: 使用缩放后的参数计算 FoV ==================
        # 注意：这里传入的是 actual_height/width 和 缩放后的 focal_length
        # 理论上 scale 分子分母会抵消，Fov 不变，但为了数值一致性，使用新数值计算
        FovY = focal2fov(focal_length_y, actual_height)
        FovX = focal2fov(focal_length_x, actual_width)

        # 加载其他特征 (保持原逻辑)
        features = torch.load(
            os.path.join(features_folder, image_name.split('.')[0] + ".pt")) if features_folder is not None else None
        
        masks = None
        if masks_folder is not None:
            mask_path_pt = os.path.join(masks_folder, image_name.split('.')[0] + ".pt")
            mask_path_png = os.path.join(masks_folder, image_name.split('.')[0] + ".png")
            if os.path.exists(mask_path_pt):
                masks = torch.load(mask_path_pt)
            elif os.path.exists(mask_path_png):
                mask_image = Image.open(mask_path_png).convert("L")
                # Resize mask to match image size if necessary
                if mask_image.size != (actual_width, actual_height):
                    mask_image = mask_image.resize((actual_width, actual_height), Image.NEAREST)
                masks = torch.from_numpy(np.array(mask_image)).float() / 255.0
                if masks.dim() == 2:
                    masks = masks.unsqueeze(0) # [1, H, W]

        mask_scales = torch.load(os.path.join(mask_scale_folder, image_name.split('.')[
            0] + ".pt")) if mask_scale_folder is not None and os.path.exists(os.path.join(mask_scale_folder, image_name.split('.')[0] + ".pt")) else None

        # ================== 修改 4: 更新 CameraInfo ==================
        # 这里的 cx, cy 已经被缩放过了
        # width, height 传入 actual_width, actual_height

        # 只有当允许主点偏移时，才传入具体的 cx, cy，否则 PyTorch3D/GaussianSplatting 可能默认使用 w/2, h/2
        final_cx = cx if allow_principle_point_shift else None
        final_cy = cy if allow_principle_point_shift else None

        cam_info = CameraInfo(
            uid=uid, R=R, T=T,
            FovY=FovY, FovX=FovX,
            image=image,
            features=features, masks=masks, mask_scales=mask_scales,
            image_path=image_path, image_name=image_name,
            width=actual_width, height=actual_height,  # 使用真实尺寸
            cx=final_cx,
            cy=final_cy
        )
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path, only_xyz=False):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors, normals = None, None
    if not only_xyz:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        if 'nx' in vertices.properties:
            normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        else:
            normals = None
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, need_features=False, need_masks=False, sample_rate = 1.0, allow_principle_point_shift = False, replica=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    print("cam_intrinsics",cam_intrinsics)

    reading_dir = "images" if images is None else images
    feature_dir = "clip_features"
    mask_dir = "masks"
    mask_scale_dir = "mask_scales"

    train_dir = os.path.join(path, reading_dir, "train")
    test_dir = os.path.join(path, reading_dir, "test")

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print("Found train/ and test/ subfolders, loading pre-split data.")

        print("Reading training cameras...")
        train_cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=train_dir, features_folder=os.path.join(path, feature_dir) if need_features else None, masks_folder=os.path.join(path, mask_dir) if need_masks else None, mask_scale_folder=os.path.join(path, mask_scale_dir) if need_masks else None, sample_rate=sample_rate, allow_principle_point_shift = allow_principle_point_shift)

        print("Reading validation cameras...")
        test_cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=test_dir, features_folder=os.path.join(path, feature_dir) if need_features else None, masks_folder=os.path.join(path, mask_dir) if need_masks else None, mask_scale_folder=os.path.join(path, mask_scale_dir) if need_masks else None, sample_rate=sample_rate, allow_principle_point_shift = allow_principle_point_shift)

        if not replica:
            train_cam_infos = sorted(train_cam_infos_unsorted, key=lambda x: x.image_name)
            test_cam_infos = sorted(test_cam_infos_unsorted, key=lambda x: x.image_name)
        else:
            train_cam_infos = sorted(train_cam_infos_unsorted, key=lambda x: int(x.image_name.split("_")[-1]))
            test_cam_infos = sorted(test_cam_infos_unsorted, key=lambda x: int(x.image_name.split("_")[-1]))

        print(f"Dataset loaded: {len(train_cam_infos)} training, {len(test_cam_infos)} validation images.")

    else:
        print("Did not find train/ and test/ subfolders. Using legacy loading logic.")
        cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), features_folder=os.path.join(path, feature_dir) if need_features else None, masks_folder=os.path.join(path, mask_dir) if need_masks else None, mask_scale_folder=os.path.join(path, mask_scale_dir) if need_masks else None, sample_rate=sample_rate, allow_principle_point_shift = allow_principle_point_shift)

        if not replica:
            cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        else:
            cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name.split("_")[-1]))

        if eval:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

# for lerf test
def readCamerasFromLerfTransforms(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents["camera_angle_x"]
        
        feature_dir = os.path.join(path, "clip_features")
        feature_dir = feature_dir if os.path.exists(feature_dir) else None
        mask_dir = os.path.join(path, "sam_masks")
        mask_dir = mask_dir if os.path.exists(mask_dir) else None
        mask_scale_dir = os.path.join(path, "mask_scales")
        mask_scale_dir = mask_scale_dir if os.path.exists(mask_scale_dir) else None
    
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # cam_name = os.path.join(path, frame["file_path"])

            tmp = np.array(frame["transform_matrix"])
            tmp_R = tmp[:3,:3]
            tmp_R = -tmp_R
            tmp_R[:,0] = -tmp_R[:,0]
            tmp[:3,:3] = tmp_R
            matrix = np.linalg.inv(tmp)
            # R = -np.transpose(matrix[:3,:3])
            # R[:,0] = -R[:,0]
            # T = -matrix[:3, 3]

            # matrix[:3,1] *= -1
            # matrix[:3,2] *= -1

            R = np.transpose(matrix[:3,:3])
            T = matrix[:3, 3]

            image_path = os.path.join(path, frame["file_path"])
            image_name = frame["file_path"].split("/")[-1].split(".")[0]
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            # fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            fovx = 2 * np.arctan(frame['w'] / (2 * frame['fl_x']))
            fovy = 2 * np.arctan(frame['h'] / (2 * frame['fl_y']))

            FovY = fovy 
            FovX = fovx

            features = torch.load(os.path.join(feature_dir, image_name.split('.')[0] + ".pt")) if feature_dir is not None else None
            masks = torch.load(os.path.join(mask_dir, image_name.split('.')[0] + ".pt")) if mask_dir is not None else None
            mask_scales = torch.load(os.path.join(mask_scale_dir, image_name.split('.')[0] + ".pt")) if mask_scale_dir is not None else None

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], features=features, masks=masks, mask_scales=mask_scales))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readLerfInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromLerfTransforms(path, "transforms.json", white_background, extension)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    test_cam_infos = []
    eval = False
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# for abo_dataset
def readCamerasFromABOTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        
        feature_dir = os.path.join(path, "clip_features_saga")
        feature_dir = feature_dir if os.path.exists(feature_dir) else None
        mask_dir = os.path.join(path, "sam_masks")
        mask_dir = mask_dir if os.path.exists(mask_dir) else None
        mask_scale_dir = os.path.join(path, "mask_scales")
        mask_scale_dir = mask_scale_dir if os.path.exists(mask_scale_dir) else None

        fovx = 2 * np.arctan(contents['w'] / (2 * contents['fl_x']))
        fovy = 2 * np.arctan(contents['h'] / (2 * contents['fl_y']))
        cx = contents['cx']
        cy = contents['cy']
    
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            tmp = np.array(frame["transform_matrix"])
            tmp_R = tmp[:3,:3]
            tmp_R = -tmp_R
            tmp_R[:,0] = -tmp_R[:,0]
            tmp[:3,:3] = tmp_R
            matrix = np.linalg.inv(tmp)

            R = np.transpose(matrix[:3,:3])
            T = matrix[:3, 3]

            image_path = os.path.join(path, frame["file_path"])
            image_name = frame["file_path"].split("/")[-1].split(".")[0]
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            FovY = fovy 
            FovX = fovx

            features = torch.load(os.path.join(feature_dir, image_name.split('.')[0] + ".pt")) if feature_dir is not None else None
            masks = torch.load(os.path.join(mask_dir, image_name.split('.')[0] + ".pt")) if mask_dir is not None else None
            mask_scales = torch.load(os.path.join(mask_scale_dir, image_name.split('.')[0] + ".pt")) if mask_scale_dir is not None else None

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, cx=cx, cy=cy,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], features=features, masks=masks, mask_scales=mask_scales))
            
    return cam_infos

def readABOInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromABOTransforms(path, "transforms.json", white_background, extension)
    test_cam_infos = []
    eval = False
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 2000
        
        # We create random points according to abo metadata
        scene_name = os.path.basename(path)
        metadata_json_path = os.path.join(path, '../../filtered_metadata.json')
        init_scale = get_initial_sacle(metadata_json_path, scene_name)
        print(f"random range scale is {init_scale}.")
        
        xyz = np.random.random((num_pts, 3)) * init_scale - init_scale/2
        # filter pcd based on mask
        xyz = filter_random_init_points(xyz, path)
        num_pts = len(xyz)
        shs = np.random.random((num_pts, 3)) / 255.0

        print(f"Generating random point cloud ({num_pts})...")
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        print(f"load init point cloud from {ply_path}...")
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def generate_novel_views(camera_list, num_novel_views=10):
    """
    在现有相机之间插值生成新的视角。

    :param camera_list: 包含现有 CameraInfo 对象的列表。
    :param num_novel_views: 要生成的新视角的数量。
    :return: 包含新生成的 CameraInfo 对象的列表。
    """
    if len(camera_list) < 2:
        print("警告: 无法生成新视角，因为相机数量少于2。")
        return []

    novel_cameras = []
    # 获取现有相机的最大 uid，以确保新 uid 不会重复
    max_uid = max(c.uid for c in camera_list) if camera_list else -1

    print(f"正在生成 {num_novel_views} 个新颖视角...")
    for i in range(num_novel_views):
        # 1. 随机选择两个不同的相机进行插值
        cam_idx1, cam_idx2 = np.random.choice(len(camera_list), 2, replace=False)
        cam1, cam2 = camera_list[cam_idx1], camera_list[cam_idx2]

        # 2. 插值旋转 (Rotation)
        # 将旋转矩阵转换为四元数
        rot1 = Rotation.from_matrix(cam1.R)
        rot2 = Rotation.from_matrix(cam2.R)

        # 创建 Slerp 插值器
        key_rots = Rotation.concatenate([rot1, rot2])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)

        # 在两个相机之间随机选择一个插值因子
        interp_factor = np.random.uniform(0.2, 0.8)  # 避免太靠近原始相机

        # 计算插值后的旋转
        interp_rotation = slerp([interp_factor])[0]
        R_new = interp_rotation.as_matrix()

        # 3. 插值位置 (Translation)
        # 首先，计算世界坐标系中的相机中心
        C1 = -np.dot(cam1.R.T, cam1.T)
        C2 = -np.dot(cam2.R.T, cam2.T)

        # 对相机中心进行线性插值 (LERP)
        C_new = (1 - interp_factor) * C1 + interp_factor * C2

        # 使用新的旋转矩阵和相机中心计算新的平移向量 T
        T_new = -np.dot(R_new, C_new)

        # 4. 创建新的 CameraInfo 对象
        # 新相机继承第一个相机的内参和图像属性（尽管图像本身不会被使用）
        new_cam_uid = max_uid + 1 + i
        novel_cam_info = CameraInfo(
            uid=new_cam_uid,
            R=R_new,
            T=T_new,
            FovY=cam1.FovY,
            FovX=cam1.FovX,
            image=cam1.image,  # 占位符
            features=None,  # 新视角没有预计算的特征
            masks=None,  # 新视角没有预计算的掩码
            mask_scales=None,
            image_path=f"novel_view_{i}",
            image_name=f"novel_view_{i}",
            width=cam1.width,
            height=cam1.height,
            cx=cam1.cx,
            cy=cam1.cy
        )
        novel_cameras.append(novel_cam_info)

    print(f"成功生成 {len(novel_cameras)} 个新视角。")
    return novel_cameras

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Lerf" : readLerfInfo,
    "ABO": readABOInfo,
}
