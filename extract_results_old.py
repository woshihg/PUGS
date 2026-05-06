import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np
from collections import defaultdict

def extract_metrics(log_dir, target_step=40):
    """提取指定 step 的指标"""
    event_file = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_file:
        return None
    
    # 按照修改时间排序，取最新的一个
    event_file.sort(key=os.path.getmtime)
    event_file = event_file[-1]
    
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    metrics = {}
    tags = ea.Tags().get('scalars', [])
    
    # 我们关注的指标关键词
    target_tags = {
        'test/loss_viewpoint - psnr': 'PSNR',
        'test/loss_viewpoint - ssim': 'SSIM',
        'test/loss_viewpoint - lpips': 'LPIPS'
    }
    
    found_any = False
    for tag in tags:
        if tag in target_tags:
            events = ea.Scalars(tag)
            # 找到最接近 target_step 的那个，或者精确匹配
            # 这里的 target_step 对应的是 iteration，如果 epoch=40 对应的是某个特定 iteration
            # 如果用户说 epoch=40，我们需要确认一下 iteration 是多少。
            # 假设用户这里的 epoch 就是 iteration (或者某种同步的 step)
            for event in events:
                if event.step == target_step:
                    metrics[target_tags[tag]] = event.value
                    found_any = True
                    break
    
    return metrics if found_any else None

def main():
    output_dir = "/home/woshihg/PycharmProjects/PUGS/output"
    # 匹配 model_2026-04-21 到 model_2026-04-27
    # 用户说是 model_2026-04-21-27 开头，可能是指 21号到27号
    # 或者就是字面意思 model_2026-04-21-27
    pattern = os.path.join(output_dir, "model_2026-04-*")
    dirs = glob.glob(pattern)
    
    # 过滤出符合条件的日期范围 (21 到 27)
    valid_dirs = []
    for d in dirs:
        name = os.path.basename(d)
        # model_2026-04-21_...
        try:
            day = int(name.split('-')[2].split('_')[0])
            if 21 <= day <= 27:
                valid_dirs.append(d)
        except:
            continue
            
    # 按时间顺序排列 (根据文件夹名)
    valid_dirs.sort()
    
    results = []
    
    # 假设每个场景按相同顺序使用了各种方法
    # 我们需要提取每个文件夹的指标
    # 既然用户提到 epoch=40，而 3DGS 通常是按 iteration 算的
    # 如果训练脚本中有 epoch 概念，通常是在 log 中体现。
    # 这里我们尝试查找 step=40 (或者尝试匹配最大的那个如果 40 不存在)
    
    for d in valid_dirs:
        name = os.path.basename(d)
        metrics = extract_metrics(d, target_step=40)
        if metrics:
            results.append({
                'Folder': name,
                'PSNR': metrics.get('PSNR', 0),
                'SSIM': metrics.get('SSIM', 0),
                'LPIPS': metrics.get('LPIPS', 0)
            })
        else:
            # 尝试打印可用的 steps 以便调试
            print(f"Warning: No metrics found at step 40 for {name}")

    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)
    print("\n--- Raw Results ---")
    print(df.to_string())
    
    # 假设“不同的场景都是按照相同的顺序使用的各种方法”
    # 意味着如果总共有 3 个场景，场景 1 的方法 A, B, C，场景 2 的 A, B, C...
    # 我们需要知道每个场景有多少个方法。
    # 用户没说具体场景数和方法数，我们假设每 N 个是一个循环。
    # 或者我们可以根据文件夹名的规律来分组（如果有的话）。
    # 这里我们先输出表格，并尝试计算每种方法的均值。
    
    # 如果不知道具体分组，我们可以猜测，或者让用户指定。
    # 常见情况是连着几组是一样的。
    
    # 尝试将结果导出为 CSV
    df.to_csv("metrics_analysis.csv", index=False)
    print("\nResults saved to metrics_analysis.csv")

if __name__ == "__main__":
    main()
