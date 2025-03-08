import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from config import tasks  # 导入配置文件


# 自定义指标函数
def custom_dc(result, reference):
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    intersection = np.count_nonzero(result & reference)
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    try:
        return 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        return 0.0


# ... 其他自定义指标函数保持不变

# 计算指标
def calculate_metrics(pred, gt):
    return {
        'dice': custom_dc(pred, gt),
        'jaccard': custom_jc(pred, gt),
        'hd95': custom_hd95(pred, gt),
        'precision': custom_precision(pred, gt),
        'recall': custom_recall(pred, gt),
        'sensitivity': custom_sensitivity(pred, gt)
    }


# 评估单个标签
def evaluate_segment(pred, gt, label):
    label_pred = (pred == label).astype(np.uint16)
    label_gt = (gt == label).astype(np.uint16)
    return calculate_metrics(label_pred, label_gt)


# 保存指标到Excel
def write_metrics_to_excel(metrics_dict, save_path, excel_name):
    with pd.ExcelWriter(os.path.join(save_path, f'{excel_name}.xlsx')) as writer:
        for label, metrics in metrics_dict.items():
            df = pd.DataFrame(metrics)
            df.to_excel(writer, sheet_name=label, index=False)


# 处理分割指标
def process_segmentation_metrics(gt_path, pred_path, save_path, excel_name, label_mapping):
    metrics_dict = {label: [] for label in label_mapping.values()}

    for file in os.listdir(gt_path):
        if file.startswith('.'):
            continue
        file_ID = file.split('.')[0]
        gt_file = os.path.join(gt_path, file)
        pred_file = os.path.join(pred_path, file)

        if os.path.exists(gt_file) and os.path.exists(pred_file):
            gt_img = sitk.ReadImage(gt_file)
            gt_arr = sitk.GetArrayFromImage(gt_img).astype(np.uint16)
            pred_img = sitk.ReadImage(pred_file)
            pred_arr = sitk.GetArrayFromImage(pred_img).astype(np.uint16)

            if gt_arr.shape != pred_arr.shape:
                print(f"Shape mismatch for {file_ID}: GT shape = {gt_arr.shape}, Pred shape = {pred_arr.shape}")
                continue

            for idx in np.unique(gt_arr):
                label_key = label_mapping.get(idx)
                if label_key and idx in np.unique(pred_arr):
                    metrics = evaluate_segment(pred_arr, gt_arr, idx)
                    metrics['file_ID'] = file_ID
                    metrics_dict[label_key].append(metrics)

    write_metrics_to_excel(metrics_dict, save_path, excel_name)


if __name__ == "__main__":
    # Label mappings can be adjusted according to the task
    label_mappings = {
        "femur": {1: 'right femur', 2: 'left femur'},
        "spine": {1: 'L5', 2: 'L4', 3: 'L3', 4: 'L2', 5: 'L1',
                  6: 'TH12', 7: 'TH11', 8: 'TH10', 9: 'TH9', 10: 'TH8'}
    }

    for task in tasks:
        label_type = "femur"  # or "spine", can be dynamically set based on task context
        process_segmentation_metrics(
            task['gt_path'], task['pred_path'], task['save_path'], task['excel_name'], label_mappings[label_type]
        )