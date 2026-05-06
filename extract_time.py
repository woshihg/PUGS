import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np

def extract_time(log_dir, target_step=4000):
    """提取到指定 step 为止的累计运行时间"""
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        return None
    
    event_files.sort(key=os.path.getmtime)
    event_file = event_files[-1]
    
    try:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        tags = ea.Tags().get('scalars', [])
        if 'iter_time' not in tags:
            return None
            
        events = ea.Scalars('iter_time')
        # 累加所有步数小于等于 target_step 的时间
        total_time = sum(e.value for e in events if e.step <= target_step)
        
        return total_time if total_time > 0 else None
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
            training_time = extract_time(d, target_step=4000)
            if training_time:
                all_results.append({
                    'Scene': scene_name,
                    'Method_ID': f"Method_{idx+1}",
                    'Folder': folder_name,
                    'Training_Time_Sec': training_time,
                    'Training_Time_Min': training_time / 60.0
                })

    if not all_results:
        print("No timing metrics (iter_time) extracted for step 4000.")
        return

    df = pd.DataFrame(all_results)
    
    # 计算每种方法的均值（跨 3 个场景）
    summary = df.groupby('Method_ID')[['Training_Time_Sec', 'Training_Time_Min']].mean().reset_index()
    
    print("\n" + "="*60)
    print("      TRAINING TIME SUMMARY (MEAN ACROSS SCENES)")
    print("="*60)
    print(summary.to_string(index=False))
    
    print("\n" + "="*60)
    print("           DETAILED TIME RESULTS (UP TO STEP 4000)")
    print("="*60)
    df_sorted = df.sort_values(by=['Method_ID', 'Scene'])
    print(df_sorted[['Method_ID', 'Scene', 'Training_Time_Min']].to_string(index=False))
    
    # 保存结果
    df_sorted.to_csv("graduation_time_analysis.csv", index=False)
    summary.to_csv("graduation_time_summary.csv", index=False)
    print("\nTiming results saved to [graduation_time_analysis.csv](graduation_time_analysis.csv) and [graduation_time_summary.csv](graduation_time_summary.csv)")

if __name__ == "__main__":
    main()
