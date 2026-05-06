import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np
from collections import defaultdict

def extract_metrics(log_dir, target_step=4000):
    """提取指定 step 的指标"""
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        return None
    
    event_files.sort(key=os.path.getmtime)
    event_file = event_files[-1]
    
    try:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        metrics = {}
        tags = ea.Tags().get('scalars', [])
        
        target_tags = {
            'test/loss_viewpoint - psnr': 'PSNR',
            'test/loss_viewpoint - ssim': 'SSIM',
            'test/loss_viewpoint - lpips': 'LPIPS'
        }
        
        found_any = False
        for tag in tags:
            if tag in target_tags:
                events = ea.Scalars(tag)
                for event in events:
                    if event.step == target_step:
                        metrics[target_tags[tag]] = event.value
                        found_any = True
                        break
        
        return metrics if found_any else None
    except Exception as e:
        print(f"Error reading {event_file}: {e}")
        return None

def main():
    # 场景路径
    base_dirs = [
        "/home/woshihg/graduation_final/scene1/output",
        "/home/woshihg/graduation_final/scene2/output",
        "/home/woshihg/graduation_final/scene3/output"
    ]
    
    all_results = []
    
    for i, base_dir in enumerate(base_dirs):
        scene_id = i + 1
        scene_name = f"scene{scene_id}"
        if not os.path.exists(base_dir):
            print(f"Warning: {base_dir} does not exist.")
            continue
            
        # 获取该场景下 model_2026-04-27 开头的目录
        dirs = [d for d in glob.glob(os.path.join(base_dir, "model_2026-04-27*")) if os.path.isdir(d)]
        # 重要：按名称排序确保方法顺序一致
        dirs.sort()
        
        scene_results = []
        for d in dirs:
            folder_name = os.path.basename(d)
            metrics = extract_metrics(d, target_step=4000)
            if metrics:
                scene_results.append({
                    'Scene': scene_name,
                    'Folder': folder_name,
                    'PSNR': metrics.get('PSNR', 0),
                    'SSIM': metrics.get('SSIM', 0),
                    'LPIPS': metrics.get('LPIPS', 0)
                })
        
        if scene_results:
            # 根据用户描述，不同场景的方法顺序完全一致，因此这里直接用索引作为 Method_${idx}
            for idx, res in enumerate(scene_results):
                res['Method_ID'] = f"Method_{idx+1}"
            all_results.extend(scene_results)

    if not all_results:
        print("No metrics extracted. Please check if training is complete and step 4000 is reached.")
        return

    df = pd.DataFrame(all_results)
    
    # 计算每种方法的均值（跨 3 个场景）
    summary = df.groupby('Method_ID')[['PSNR', 'SSIM', 'LPIPS']].mean().reset_index()
    
    print("\n" + "="*60)
    print("      SUMMARY BY METHOD (MEAN ACROSS SCENES)")
    print("="*60)
    print(summary.to_string(index=False))
    
    print("\n" + "="*60)
    print("              DETAILED RESULTS (STEP 4000)")
    print("="*60)
    # 按方法和场景排序
    df_sorted = df.sort_values(by=['Method_ID', 'Scene'])
    print(df_sorted[['Method_ID', 'Scene', 'PSNR', 'SSIM', 'LPIPS']].to_string(index=False))
    
    # 保存结果
    df_sorted.to_csv("graduation_metrics_analysis.csv", index=False)
    summary.to_csv("graduation_summary.csv", index=False)
    print("\nResults saved to [graduation_metrics_analysis.csv](graduation_metrics_analysis.csv) and [graduation_summary.csv](graduation_summary.csv)")

if __name__ == "__main__":
    main()
