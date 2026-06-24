import os
import numpy as np

def read_bbox_txt(file_path):

    bboxes = []
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            vals = line.strip().replace(',', ' ').split()
            if len(vals) == 4:
                x, y, w, h = map(float, vals)
                bboxes.append([x, y, w, h])
            else:
                bboxes.append(list(map(float, vals)))
    return np.array(bboxes)

def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / (boxAArea + boxBArea - interArea + 1e-8)

def center_error(boxA, boxB):

    cxA, cyA = boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2
    cxB, cyB = boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2
    return np.sqrt((cxA - cxB)**2 + (cyA - cyB)**2)

def compute_metrics(pred_root, gt_root, precision_threshold=20, success_threshold=0.5, lost_iou_threshold=0.0):
    """
    IoU, Precision, SuccessRate, EAO
    """
    seqs = sorted([d for d in os.listdir(gt_root) if os.path.isdir(os.path.join(gt_root, d))])
    all_iou, all_precision, all_success = [], [], []
    all_eao_values = []

    for seq in seqs:
        pred_file = os.path.join(pred_root, f"{seq}.txt")
        gt_file = os.path.join(gt_root, seq, "init.txt")

        if not os.path.exists(pred_file) or not os.path.exists(gt_file):
            print(f"no file {seq}")
            continue

        pred_bboxes = read_bbox_txt(pred_file)
        gt_bboxes = read_bbox_txt(gt_file)

        if len(pred_bboxes) != len(gt_bboxes):
            print(f"[warnning] {seq}")
            continue

        seq_iou, seq_precision, seq_success, seq_eao = [], [], [], []
        for p, g in zip(pred_bboxes, gt_bboxes):
            f_iou = iou(p, g)
            seq_iou.append(f_iou)
            seq_precision.append(1 if center_error(p, g) <= precision_threshold else 0)
            seq_success.append(1 if f_iou >= success_threshold else 0)
            seq_eao.append(f_iou if f_iou > lost_iou_threshold else 0)

        all_iou.extend(seq_iou)
        all_precision.extend(seq_precision)
        all_success.extend(seq_success)
        all_eao_values.append(seq_eao)


    metrics = {
        "IoU": np.mean(all_iou),
        "Precision": np.mean(all_precision),
        "SuccessRate": np.mean(all_success)
    }


    max_len = max(len(s) for s in all_eao_values)
    eao_list = []
    for l in range(1, max_len+1):
        overlaps = []
        for seq in all_eao_values:
            if len(seq) >= l:
                for start in range(len(seq)-l+1):
                    overlaps.append(np.mean(seq[start:start+l]))
        if overlaps:
            eao_list.append(np.mean(overlaps))
    metrics["EAO"] = np.mean(eao_list) if eao_list else 0.0

    return metrics


if __name__ == "__main__":
    pred_root = "/data/Newdisk2/linxiuhao/old_project/object_tracing_project/ViPT-clear/RGBT_workspace/results/VOT2020/deep_rgbt"
    gt_root = "/data/Newdisk2/linxiuhao/old_project/object_tracing_project/ViPT-clear/data/VOT2020"

    metrics = compute_metrics(pred_root, gt_root)
    print("=== Tracking Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
