import cv2
import os
import argparse
import math
from tqdm import tqdm

def extract_frames(video_path, output_dir, total_frames=None, fps=None):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = video_total_frames / video_fps if video_fps > 0 else 0

    print(f"Video Info: \n- FPS: {video_fps:.2f}\n- Total Frames: {video_total_frames}\n- Duration: {video_duration:.2f}s")

    # Determine frame indices to extract
    frame_indices = []
    
    if total_frames is not None:
        if total_frames > video_total_frames:
            print(f"Warning: Requested total_frames ({total_frames}) exceeds video total frames ({video_total_frames}). Clamping to {video_total_frames}.")
            total_frames = video_total_frames
        
        # Linearly space the indices over the whole video
        # Using linspace equivalent to evenly sample
        step = video_total_frames / total_frames
        frame_indices = [int(i * step) for i in range(total_frames)]
        
    elif fps is not None:
        if fps > video_fps:
            print(f"Warning: Requested fps ({fps}) exceeds video fps ({video_fps}). Clamping to {video_fps}.")
            fps = video_fps
            
        step = video_fps / fps
        num_frames = int(video_duration * fps)
        frame_indices = [int(i * step) for i in range(num_frames)]
        
    else:
        print("Error: You must specify either --total_frames or --fps")
        cap.release()
        return

    # Ensure no out of bounds indices just in case
    frame_indices = [idx for idx in frame_indices if idx < video_total_frames]
    
    print(f"\nStarting extraction of {len(frame_indices)} frames...")

    count = 0
    current_index_pos = 0
    
    with tqdm(total=len(frame_indices), desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_index_pos < len(frame_indices) and count == frame_indices[current_index_pos]:
                # Save frame
                filename = os.path.join(output_dir, f"{current_index_pos:04d}.png")
                cv2.imwrite(filename, frame)
                
                current_index_pos += 1
                pbar.update(1)
                
            count += 1
            
            # If we've processed all needed frames, stop early
            if current_index_pos >= len(frame_indices):
                break

    cap.release()
    print(f"\nSuccessfully extracted {current_index_pos} frames to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from an MP4 video uniformly.")
    parser.add_argument("video_path", type=str, help="Path to the input video (.mp4)")
    parser.add_argument("output_dir", type=str, help="Directory to save the output images")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-n", "--total_frames", type=int, help="Total number of frames to sample")
    group.add_argument("-m", "--fps", type=float, help="Number of frames to sample per second")

    args = parser.parse_args()

    extract_frames(args.video_path, args.output_dir, args.total_frames, args.fps)

if __name__ == "__main__":
    main()
