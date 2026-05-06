import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np

def extract_miou(log_dir, target_step=4000):
    """提取指定 step 的 mIoU 指标"""
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
        
        # 目标标签关注 test/loss_viewpoint - miou 和 train_loss_patches/miou
        target_tags = {
            'test/loss_viewpoint - miou': 'Test_mIoU',
            'train_loss_patches/miou': 'Train_mIoU'
        }
        
        found_any = False
        for tag in tags:
            if tag in target_tags:
                events = ea.Scalars(tag)
                # 查找指定的 step (step=4000)
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
            continue
            
        dirs = [d for d in glob.glob(os.path.join(base_dir, "model_2026-04-27*")) if os.path.isdir(d)]
        dirs.sort()
        
        for idx, d in enumerate(dirs):
            folder_name = os.path.basename(d)
            metrics = extract_miou(d, target_step=4000)
            if metrics:
                all_results.append({
                    'Scene': scene_name,
                    'Method_ID': f"Method_{idx+1}",
                    'Folder': folder_name,
                    'Test_mIoU': metrics.get('Test_mIoU', 0),
                    'Train_mIoU': metrics.get('Train_mIoU', 0)
                })

    if not all_results:
        print("No mIoU metrics extracted for step 4000.")
        return

    df = pd.DataFrame(all_results)
    
    # 计算每种方法的均值（跨 3 个场景）
    summary = df.groupby('Method_ID')[['Test_mIoU', 'Train_mIoU']].mean().reset_index()
    
    print("\n" + "="*60)
    print("      mIoU SUMMARY BY METHOD (MEAN ACROSS SCENES)")
    print("="*60)
    print(summary.to_string(index=False))
    
    print("\n" + "="*60)
    print("           DETAILED mIoU RESULTS (STEP 4000)")
    print("="*60)
    df_sorted = df.sort_values(by=['Method_ID', 'Scene'])
    print(df_sorted[['Method_ID', 'Scene', 'Test_mIoU']].to_string(index=False))
    
    # 保存结果
    df_sorted.to_csv("graduation_miou_analysis.csv", index=False)
    summary.to_csv("graduation_miou_summary.csv", index=False)
    print("\nmIoU results saved to [graduation_miou_analysis.csv](graduation_miou_analysis.csv) and [graduation_miou_summary.csv](graduation_miou_summary.csv)")

if __name__ == "__main__":
    main()
