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
import time
from pathlib import Path
# 替换为你的代理地址，注意：
# 1. 即使是 https 协议，key 也建议全大写
# 2. 如果是本地代理，通常是 127.0.0.1:端口号
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

import sys

import torch
import uuid

import numpy as np

from random import randint
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.loss_utils import l1_loss, ssim, get_img_grad_weight, zero_one_loss, LPIPS
from utils.general_utils import safe_state
from utils.image_utils import psnr, erode
from utils.difix_utils import CameraPoseInterpolator
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from scene.dataset_readers import CameraInfo
from gaussian_renderer import render, network_gui, render_at_plane
from Difix3D.src.pipeline_difix import DifixPipeline
from torchvision.transforms import ToPILImage
from diffusers.utils import load_image

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, pretrained_gaussians, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    if pretrained_gaussians:
        gaussians_path = os.path.join(dataset.source_path, "0000.ply")
        gaussians.load_ply(gaussians_path, rate=dataset.gaussian_load_rate)
        print(f"Loaded pretrained gaussians from {gaussians_path}")
    gaussians.training_setup(opt)

    if pipe.use_segmentation:
        num_classes = dataset.num_classes
        print("Num classes: ",num_classes)
        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
        classifier.cuda()

    difix = None
    if pipe.use_fix:
        difix = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
        difix.to("cuda")

    # Initialize LPIPS metric
    lpips_metric = LPIPS(net='vgg').to('cuda')

    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        # Alternate between training and novel views
        if iteration % 2 == 1 and scene.getNovelCameras():
            novel_viewpoint_stack = scene.getNovelCameras().copy()
            viewpoint_cam = novel_viewpoint_stack.pop(randint(0, len(novel_viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_at_plane(viewpoint_cam, gaussians, pipe, background,
                                    return_plane=iteration > opt.single_view_weight_from_iter,
                                    return_depth_normal=iteration > opt.single_view_weight_from_iter)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if iteration in testing_iterations:
            # Run Difix cycle to generate and add new training data
            if pipe.use_fix:
                run_difix_cycle(scene, iteration, difix, render, (pipe, background))


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        # Apply mask if available
        if viewpoint_cam.original_masks is not None:
            mask = viewpoint_cam.original_masks.cuda()
            # Ensure mask has same resolution as image
            if mask.shape[1:] != image.shape[1:]:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=image.shape[1:], mode='nearest').squeeze(0)
            
            # Masked L1 and SSIM
            # We use a more robust masked loss by only averaging over the mask
            mask_sum = mask.sum() + 1e-8
            Ll1 = (torch.abs(image - gt_image) * mask).sum() / mask_sum
            # For SSIM, we mask the images. SSIM is window-based, so masking the input is a common approximation.
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image * mask, gt_image * mask))
        else:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # Geometry_Loss
        if iteration > opt.single_view_weight_from_iter:
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 5
            image_weight = erode(image_weight[None,None]).squeeze()

            if viewpoint_cam.original_masks is not None:
                mask = viewpoint_cam.original_masks.cuda()
                if mask.shape[1:] != image_weight.shape:
                    mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=image_weight.shape, mode='nearest').squeeze()
                image_weight = image_weight * mask.squeeze()

            normal_loss = (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            # loss += (opt.lambda_single_view * normal_loss)
            
        # Sparse Loss
        alpha = render_pkg["alpha"]
        if viewpoint_cam.original_masks is not None:
            mask = viewpoint_cam.original_masks.cuda()
            if mask.shape[1:] != alpha.shape[1:]:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=alpha.shape[1:], mode='nearest').squeeze(0)
            
            mask_bool = mask[0] > 0.5
            if mask_bool.any():
                # Index alpha using the 2D mask. If alpha is [C, H, W], result is [C, N]
                loss_01 = zero_one_loss(alpha[..., mask_bool])
            else:
                loss_01 = torch.tensor(0.0, device="cuda")
        else:
            loss_01 = zero_one_loss(alpha)
        loss += (opt.lambda_zero_one * loss_01)

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Compute elapsed time for this iteration (in seconds) using CUDA events
            try:
                # Ensure all CUDA work is finished before measuring
                torch.cuda.synchronize()
                elapsed_ms = iter_start.elapsed_time(iter_end)
                elapsed = elapsed_ms / 1000.0
            except Exception:
                # Fallback if CUDA timing not available
                elapsed = 0.0

            # Call training_report safely so failures don't interrupt training
            try:
                # renderFunc = render, renderArgs = (pipe, background)
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                                testing_iterations, scene, render, (pipe, background), lpips_metric)
            except Exception as e:
                # Don't interrupt training if reporting fails
                print(f"[WARN] training_report failed at iter {iteration}: {e}")

            # Log and save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def _resolve_image_path(dataset_root, cam):
    # if image_name present, try to build candidate paths relative to dataset/source_path
    image_name = getattr(cam, 'image_name', None)

    if image_name:
        # image_name may already be a relative path or include folders
        for suffix in ['.png', '.jpg', '.jpeg']:
            os.path.join(dataset_root, 'images', 'train', image_name + suffix)
            if os.path.exists(os.path.join(dataset_root, 'images', 'train', image_name + suffix)):
                return os.path.join(dataset_root, 'images', 'train', image_name + suffix)
    # Not found
    return None

def run_difix_cycle(scene: Scene, iteration: int, difix_pipe, renderFunc, renderArgs, shift_distance=0.1):
    """
    Generates novel views by shifting train poses towards test poses, fixes them
    using Difix, and adds them to a dedicated set of novel cameras in the scene.
    """
    if not difix_pipe:
        return

    train_cameras = scene.getTrainCameras()
    novel_cameras = scene.getNovelCameras()
    all_source_cameras = train_cameras + novel_cameras

    test_cameras = scene.getTestCameras()

    if not train_cameras or not test_cameras:
        print(f"[ITER {iteration}] No training or testing cameras found, skipping novel view generation.")
        return

    # Create iteration-specific directories for saving outputs
    iter_dir = os.path.join(scene.model_path, "fix", f"iter_{iteration}")
    render_dir = os.path.join(iter_dir, "render_novel")
    fixed_dir = os.path.join(iter_dir, "fixed_novel")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(fixed_dir, exist_ok=True)

    # --- Step 1: Generate Novel View Poses ---
    pose_interpolator = CameraPoseInterpolator()
    # 插值的时候使用的是相机在世界坐标系下的位姿
    train_poses = np.array([w2c_to_c2w(cam.R, cam.T) for cam in train_cameras])
    all_source_poses = np.array([w2c_to_c2w(cam.R, cam.T) for cam in all_source_cameras])
    test_poses = np.array([w2c_to_c2w(cam.R, cam.T) for cam in test_cameras])

    # Generate novel poses by shifting from training poses towards testing poses
    novel_poses = pose_interpolator.shift_poses(all_source_poses, test_poses, distance=shift_distance)

    # --- Step 2: Create Novel Cameras and Render Them ---
    to_pil = ToPILImage()
    new_cameras_to_add = []
    render_items = []

    print(f"\n[ITER {iteration}] Step 1: Rendering {len(novel_poses)} novel views...")
    for i, novel_pose in enumerate(tqdm(novel_poses, desc="Rendering Novel Views")):
        # Find the nearest training camera to use as a template for intrinsics
        assignments = pose_interpolator.find_nearest_assignments(train_poses, [novel_pose])
        template_cam = train_cameras[assignments[0]]
        # Create a new CameraInfo for the novel view
        R = novel_pose[:3, :3].transpose()
        T = -R @ novel_pose[:3, 3]
        novel_cam_info = CameraInfo(
            uid=-1,  # UID will be assigned later
            R=R,
            T=T,
            FovY=template_cam.FoVy,
            FovX=template_cam.FoVx,
            image=None,  # No ground truth image
            features=None, masks=None, mask_scales=None,
            image_path=None,
            image_name=test_cameras[i].image_name,
            width=template_cam.image_width,
            height=template_cam.image_height,
            cx=getattr(template_cam, "cx", None),
            cy=getattr(template_cam, "cy", None),
        )

        # Create a full Camera object for rendering
        from scene.cameras import Camera
        novel_camera = Camera(
            uid = None, colmap_id=novel_cam_info.uid, R=novel_cam_info.R, T=novel_cam_info.T,
            FoVx=novel_cam_info.FovX, FoVy=novel_cam_info.FovY,
            image_width=novel_cam_info.width, image_height=novel_cam_info.height,
            image_name=novel_cam_info.image_name, image=None,
            data_device="cuda"
        )

        # Render the novel view
        with torch.no_grad():
            render_pkg = renderFunc(novel_camera, scene.gaussians, *renderArgs)
        rendered_image_tensor = torch.clamp(render_pkg["render"], 0.0, 1.0)
        rendered_image_pil = to_pil(rendered_image_tensor.cpu())

        render_path = os.path.join(render_dir, f"{novel_camera.image_name}.png")
        rendered_image_pil.save(render_path)
        render_items.append({"viewpoint": novel_camera, "render_path": render_path, "novel_pose": novel_pose})

    # --- Step 3: Apply Difix to Rendered Novel Views ---
    train_image_paths = [_resolve_image_path(scene.source_path, cam) for cam in train_cameras]

    print(f"\n[ITER {iteration}] Step 2: Applying Difix to {len(render_items)} rendered novel views...")
    for item in tqdm(render_items, desc="Applying Difix to Novel Views"):
        viewpoint = item["viewpoint"]
        render_path = item["render_path"]
        novel_pose = item["novel_pose"]

        rendered_image_pil = load_image(render_path)

        # Find nearest training camera for reference
        assignments = pose_interpolator.find_nearest_assignments(train_poses, [novel_pose])
        ref_image_path = train_image_paths[assignments[0]]

        if ref_image_path is None:
            print(f"[WARN] Skipping Difix for {viewpoint.image_name} due to missing reference image.")
            continue

        reference_image_pil = load_image(ref_image_path)
        # 将reference_image_pil 分辨率和 rendered_image_pil 对齐
        reference_image_pil = reference_image_pil.resize(rendered_image_pil.size)

        # Apply Difix
        with torch.no_grad():
            result = difix_pipe(prompt="remove degradation",
                                image=rendered_image_pil,
                                ref_image=reference_image_pil,
                                num_inference_steps=1,
                                timesteps=[199],
                                guidance_scale=0.0)
            fixed_pil = result.images[0].resize(rendered_image_pil.size)

        render_image_name = f"{viewpoint.image_name}_{iteration}"
        fixed_path = os.path.join(fixed_dir, f"{render_image_name}.png")
        fixed_pil.save(fixed_path)

        # --- Step 4: Prepare CameraInfo to be added to the novel set ---
        new_uid = scene.get_max_uid() + 1
        new_cam_info = CameraInfo(
            uid=new_uid,
            R=viewpoint.R,
            T=viewpoint.T,
            FovY=viewpoint.FoVy,
            FovX=viewpoint.FoVx,
            image=fixed_pil,
            features=None, masks=None, mask_scales=None,
            image_path=fixed_path,
            image_name=render_image_name,
            width=viewpoint.image_width,
            height=viewpoint.image_height,
            cx=getattr(viewpoint, "cx", None),
            cy=getattr(viewpoint, "cy", None),
        )
        new_cameras_to_add.append(new_cam_info)

    # Add the newly generated and fixed views to the scene's novel camera set
    if new_cameras_to_add:
        scene.add_novel_cameras(new_cameras_to_add)
        print(f"\n[ITER {iteration}] Added {len(new_cameras_to_add)} new views to the novel camera set.")

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, lpips_metric):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_metric(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def w2c_to_c2w(R,t):
    # 2. 计算逆旋转（即 R 的转置）
    R_inv = R.T

    # 3. 计算新的平移向量
    t_inv = -np.dot(R_inv, t)

    # 4. 组装成新的 C2W 矩阵
    c2w_matrix = np.eye(4)
    c2w_matrix[:3, :3] = R_inv
    c2w_matrix[:3, 3] = t_inv

    return c2w_matrix

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=np.random.randint(10000, 20000))
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 4_500,
                                                                           5_000, 5_500, 6_000, 6_500, 7_000, 7_500, 8_000, 8_500, 9_000, 9_500, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[500, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 4_500,
                                                                           5_000, 5_500, 6_000, 6_500, 7_000, 7_500, 8_000, 8_500, 9_000, 9_500, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--pretrained_gaussians", action="store_true", help="Use Pretrained Gaussians")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # Check if masks directory exists and set need_masks accordingly
    if os.path.exists(os.path.join(args.source_path, "masks")):
        args.need_masks = True
        print("Found masks directory, enabling mask-based optimization.")
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.pretrained_gaussians, args.debug_from)

    # All done
    print("\nTraining complete.")
