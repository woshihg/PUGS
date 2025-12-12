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
from utils.loss_utils import l1_loss, ssim, get_img_grad_weight, zero_one_loss
from utils.general_utils import safe_state
from utils.image_utils import psnr, erode
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
        # 保存采样后的ply文件以供检查
        # sampled_ply_path = os.path.join(scene.model_path, "sampled_pretrained_gaussians.ply")
        # gaussians.save_ply(sampled_ply_path)
        # print(f"Saved sampled pretrained gaussians to {sampled_ply_path}")
    gaussians.training_setup(opt)
    difix = None
    if pipe.use_fix:
        difix = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)

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
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # Geometry_Loss
        if iteration > opt.single_view_weight_from_iter:
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 5
            image_weight = erode(image_weight[None,None]).squeeze()

            normal_loss = (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            loss += (opt.lambda_single_view * normal_loss)
            
        # Sparse Loss
        loss_01 = zero_one_loss(render_pkg["alpha"])
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
                                testing_iterations, scene, render, (pipe, background))
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

def run_difix_cycle(scene: Scene, iteration: int, difix_pipe, renderFunc, renderArgs):
    """
    Renders all validation views and saves them. Then, loads the saved images,
    fixes them one by one using Difix, and adds the results to the training set.
    """
    if not difix_pipe:
        return

    val_cameras = scene.getTestCameras()
    if not val_cameras:
        print(f"[ITER {iteration}] No validation cameras found, skipping Difix cycle.")
        return

    render_dir = os.path.join(scene.model_path, "fix", "render")
    fixed_dir = os.path.join(scene.model_path, "fix", "fixed")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(fixed_dir, exist_ok=True)

    to_pil = ToPILImage()
    new_cameras_to_add = []
    render_items = []

    # Step 1: Render all validation views and save them to disk
    print(f"\n[ITER {iteration}] Step 1: Rendering {len(val_cameras)} validation views...")
    for viewpoint in tqdm(val_cameras, desc="Rendering Views"):
        with torch.no_grad():
            render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
        rendered_image_tensor = torch.clamp(render_pkg["render"], 0.0, 1.0)
        rendered_image_pil = to_pil(rendered_image_tensor.cpu())

        render_path = os.path.join(render_dir, f"iter_{iteration}_{viewpoint.image_name}.png")
        rendered_image_pil.save(render_path)
        render_items.append({"viewpoint": viewpoint, "render_path": render_path})

    # Step 2: Load rendered images, apply Difix, and replace/add to training set
    print(f"\n[ITER {iteration}] Step 2: Applying Difix to {len(render_items)} rendered views...")
    for item in tqdm(render_items, desc="Applying Difix"):
        viewpoint = item["viewpoint"]
        render_path = item["render_path"]

        # Load the saved rendered image
        rendered_image_pil = load_image(render_path)

        # Call Difix on the single loaded image
        try:
            with torch.no_grad():
                result = difix_pipe(prompt="remove degradation",
                                    image=rendered_image_pil,
                                    num_inference_steps=1,
                                    timesteps=[199],
                                    guidance_scale=0.0)
                fixed_pil = result.images[0]
                print(f"\n finish fix {render_path}")
        except Exception as e:
            print(f"[WARN] Difix failed on view {getattr(viewpoint, 'image_name', 'unknown')}: {e}")
            continue  # Skip this image if Difix fails

        # Save the fixed image
        fixed_path = os.path.join(fixed_dir, f"iter_{iteration}_{viewpoint.image_name}.png")
        fixed_pil.save(fixed_path)

        # Determine uid: prefer to reuse existing train camera uid if same view exists
        existing_uid = None
        try:
            for cam in scene.getTrainCameras():
                if hasattr(cam, "image_name") and cam.image_name == viewpoint.image_name:
                    existing_uid = getattr(cam, "uid", None)
                    break
        except Exception:
            existing_uid = None

        new_uid = existing_uid if existing_uid is not None else (scene.get_max_uid() + 1)

        # Construct new CameraInfo (reuse viewpoint intrinsics/poses)
        new_cam_info = CameraInfo(
            uid=new_uid,
            R=viewpoint.R,
            T=viewpoint.T,
            FovY=getattr(viewpoint, "FoVy", getattr(viewpoint, "FovY", None)),
            FovX=getattr(viewpoint, "FoVx", getattr(viewpoint, "FovX", None)),
            image=fixed_pil,  # Store PIL image directly
            features=None, masks=None, mask_scales=None,
            image_path=fixed_path,
            image_name=viewpoint.image_name,
            width=getattr(viewpoint, "image_width", getattr(viewpoint, "width", None)),
            height=getattr(viewpoint, "image_height", getattr(viewpoint, "height", None)),
            cx = getattr(viewpoint, "Cx", getattr(viewpoint, "cx", None)),
            cy = getattr(viewpoint, "Cy", getattr(viewpoint, "cy", None)),
        )

        # Replace existing train camera with same view or add if not present
        scene.add_train_cameras(new_cam_info)

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
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
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_0, 2_000, 3_000, 5_000, 7_000, 10_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 2_000, 3_000, 5_000, 7_000, 10_000, 20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--pretrained_gaussians", action="store_true", help="Use Pretrained Gaussians")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
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
