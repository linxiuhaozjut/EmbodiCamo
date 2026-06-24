import os
import cv2
import sys
from os.path import join, isdir, dirname
import numpy as np
import argparse
import multiprocessing
import torch
import time

from lib.train.dataset.depth_utils import get_x_frame
from lib.test.tracker.ostrack import OSTrack
from lib.test.tracker.vipt import ViPTTrack
import lib.test.parameter.vipt as rgbt_prompt_params

prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)


def gray_grad(grad):
    # Convert the gradient to grayscale and replicate it across 3 channels.
    gray = grad.mean(axis=2, keepdims=True)
    gray3 = np.repeat(gray, 3, axis=2)
    return gray3


def process_got_with_patch(
        patch_v, patch_i,
        data_root="./data/GTOT",
        mask_root="./mask/GTOT",
        out_root="./adv/GTOT"):
    """Apply external visible and infrared patches to the GTOT dataset."""
    print("Start processing the GTOT dataset with external patches...")

    # Ensure the patches are numpy arrays.
    patch_v = np.asarray(patch_v, dtype=np.float32)
    patch_i = np.asarray(patch_i, dtype=np.float32)

    for seq in sorted(os.listdir(data_root)):
        seq_path = os.path.join(data_root, seq)
        if not os.path.isdir(seq_path):
            continue

        print(f"Processing sequence: {seq}")

        v_dir = os.path.join(seq_path, "v")
        i_dir = os.path.join(seq_path, "i")
        if not (os.path.isdir(v_dir) and os.path.isdir(i_dir)):
            print(f"{seq} is missing the v/ or i/ folder, skipping.")
            continue

        out_v_dir = os.path.join(out_root, seq, "v")
        out_i_dir = os.path.join(out_root, seq, "i")
        os.makedirs(out_v_dir, exist_ok=True)
        os.makedirs(out_i_dir, exist_ok=True)

        gt_path = os.path.join(seq_path, "init.txt")
        if not os.path.exists(gt_path):
            print(f"{seq} is missing init.txt.")
            continue

        gtb = np.loadtxt(gt_path)
        if gtb.ndim == 1:
            gtb = gtb[None, :]

        v_list = sorted([p for p in os.listdir(v_dir) if p.lower().endswith(("png", "bmp"))])
        i_list = sorted([p for p in os.listdir(i_dir) if p.lower().endswith(("png", "bmp"))])

        if len(v_list) != len(gtb):
            print(f"{seq} has inconsistent image and GT counts, skipping.")
            continue

        # Fix the patch region using the first-frame target box.
        x0, y0, w0, h0 = map(int, gtb[0])
        cx0 = x0 + w0 // 2
        cy0 = y0 + h0 // 2
        size0 = int(max(w0, h0) * 1.25)

        pv0 = cv2.resize(patch_v, (size0, size0), interpolation=cv2.INTER_LINEAR)
        pi0 = cv2.resize(patch_i, (size0, size0), interpolation=cv2.INTER_LINEAR)

        for idx, (vf, inf) in enumerate(zip(v_list, i_list)):
            img_v = cv2.imread(os.path.join(v_dir, vf))
            img_i = cv2.imread(os.path.join(i_dir, inf))
            H, W = img_v.shape[:2]

            x1 = cx0 - size0 // 2
            y1 = cy0 - size0 // 2
            x2 = x1 + size0
            y2 = y1 + size0

            # Clamp the patch region to image boundaries.
            px1 = max(0, -x1)
            py1 = max(0, -y1)
            px2 = size0 - max(0, x2 - W)
            py2 = size0 - max(0, y2 - H)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)

            pv_crop = pv0[py1:py2, px1:px2]
            pi_crop = pi0[py1:py2, px1:px2]

            mask_name = vf.replace(".png", "_mask.png").replace(".bmp", "_mask.bmp")
            mask_path = os.path.join(mask_root, seq, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Mask not found: {mask_path}")
                continue

            mask_bin = (mask > 125).astype(np.uint8)
            mask_crop = mask_bin[y1:y2, x1:x2]
            inv_mask = (1 - mask_crop)[:, :, None]

            img_v[y1:y2, x1:x2] = img_v[y1:y2, x1:x2] * mask_crop[:, :, None] + pv_crop * inv_mask
            img_i[y1:y2, x1:x2] = img_i[y1:y2, x1:x2] * mask_crop[:, :, None] + pi_crop * inv_mask

            cv2.imwrite(os.path.join(out_v_dir, vf), img_v)
            cv2.imwrite(os.path.join(out_i_dir, inf), img_i)

    print("Finished processing all sequences with external patches.")


def compute_iou(box1, box2):
    """Compute IoU for boxes in (x, y, w, h) format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter

    if union == 0:
        return 0
    return inter / union


def genConfig(seq_path, set_type):
    if set_type == 'VOT2020':
        RGB_img_list = sorted([seq_path + '/vi/' + p for p in os.listdir(seq_path + '/vi')
                               if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted([seq_path + '/ir/' + p for p in os.listdir(seq_path + '/ir')
                             if os.path.splitext(p)[1] == '.jpg'])

        RGB_gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')

    elif set_type == 'GTOT':
        RGB_img_list = sorted(
            [seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if p.lower().endswith(('.png', '.bmp'))]
        )
        T_img_list = sorted(
            [seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if p.lower().endswith(('.png', '.bmp'))]
        )

        RGB_gt = np.loadtxt(seq_path + '/groundTruth_v.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/groundTruth_i.txt', delimiter=' ')

        x_min = np.min(RGB_gt[:, [0, 2]], axis=1)[:, None]
        y_min = np.min(RGB_gt[:, [1, 3]], axis=1)[:, None]
        x_max = np.max(RGB_gt[:, [0, 2]], axis=1)[:, None]
        y_max = np.max(RGB_gt[:, [1, 3]], axis=1)[:, None]
        RGB_gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

        x_min = np.min(T_gt[:, [0, 2]], axis=1)[:, None]
        y_min = np.min(T_gt[:, [1, 3]], axis=1)[:, None]
        x_max = np.max(T_gt[:, [0, 2]], axis=1)[:, None]
        y_max = np.max(T_gt[:, [1, 3]], axis=1)[:, None]
        T_gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    elif set_type == 'LasHeR':
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible')
                               if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared')
                             if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif 'VTUAV' in set_type:
        RGB_img_list = sorted([seq_path + '/rgb/' + p for p in os.listdir(seq_path + '/rgb')
                               if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/ir/' + p for p in os.listdir(seq_path + '/ir')
                             if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/rgb.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/ir.txt', delimiter=' ')

    return RGB_img_list, T_img_list, RGB_gt, T_gt


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=300, debug=0, script_name='prompt'):
    if 'VTUAV' in dataset_name:
        seq_txt = seq_name.split('/')[1]
    else:
        seq_txt = seq_name

    save_name = '{}'.format(yaml_name)
    save_path = f'./RGBT_workspace/results/{dataset_name}/' + save_name + '/' + seq_txt + '.txt'
    save_folder = f'./RGBT_workspace/results/{dataset_name}/' + save_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    if script_name == 'vipt':
        params = rgbt_prompt_params.parameters(yaml_name, epoch)
        mmtrack = ViPTTrack(params)
        tracker = ViPT_RGBT(tracker=mmtrack)

    seq_path = seq_home + '/' + seq_name
    print('Process sequence: ' + seq_name)
    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    gt = np.copy(RGB_gt)
    toc = 0
    iou_sum = 0

    for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
        tic = cv2.getTickCount()
        if frame_idx == 0:
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb'))
            tracker.initialize(image, RGB_gt[0].tolist())

            x, y, w, h = map(int, result[0])
            imagergb = image[:, :, :3].copy()
            imaget = image[:, :, 3:6].copy()
            cv2.rectangle(imagergb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imaget, (x, y), (x + w, y + h), (0, 255, 0), 2)

            os.makedirs(os.path.join(save_folder, "vision", seq_txt, "v"), exist_ok=True)
            os.makedirs(os.path.join(save_folder, "vision", seq_txt, "i"), exist_ok=True)
            v_path = os.path.join(save_folder, "vision", seq_txt, "v", f"{frame_idx:05d}.jpg")
            i_path = os.path.join(save_folder, "vision", seq_txt, "i", f"{frame_idx:05d}.jpg")
            cv2.imwrite(v_path, imagergb)
            cv2.imwrite(i_path, imaget)
        elif frame_idx > 0:
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb'))
            region, confidence = tracker.track(image)
            result[frame_idx] = np.array(region)

            iou = compute_iou(result[frame_idx], gt[frame_idx])
            print(seq_txt, "frame:", frame_idx, "pred box:", result[frame_idx], "gt box:", gt[frame_idx], "iou:", iou)
            iou_sum += iou

            x, y, w, h = map(int, result[frame_idx])
            xg, yg, wg, hg = map(int, gt[frame_idx])
            imagergb = image[:, :, :3].copy()
            imaget = image[:, :, 3:6].copy()
            cv2.rectangle(imagergb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imagergb, (xg, yg), (xg + wg, yg + hg), (0, 0, 255), 2)
            cv2.rectangle(imaget, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imaget, (xg, yg), (xg + wg, yg + hg), (0, 0, 255), 2)

            os.makedirs(os.path.join(save_folder, "vision", seq_txt, "v"), exist_ok=True)
            os.makedirs(os.path.join(save_folder, "vision", seq_txt, "i"), exist_ok=True)
            v_path = os.path.join(save_folder, "vision", seq_txt, "v", f"{frame_idx:05d}.jpg")
            i_path = os.path.join(save_folder, "vision", seq_txt, "i", f"{frame_idx:05d}.jpg")
            cv2.imwrite(v_path, imagergb)
            cv2.imwrite(i_path, imaget)

        toc += cv2.getTickCount() - tic

    toc /= cv2.getTickFrequency()
    avg_iou = iou_sum / frame_idx
    print(seq_txt, "mean iou:", avg_iou)
    if not debug:
        np.savetxt(save_path, result)
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))

    return avg_iou


class ViPT_RGBT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        init_info = {'init_bbox': list(gt_bbox_np)}  # Input must be in (x, y, w, h) format.
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, default='prompt', help='Name of tracking method(ostrack, prompt, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='ostrack_ce_ep60_prompt_iv21b_wofovea_8_onlylasher_2xa100_rgbt', help='Name of tracking method.')
    parser.add_argument('--dataset_name', type=str, default='LasHeR', help='Name of dataset (GTOT,RGBT234,LasHeR,VTUAVST,VTUAVLT).')
    parser.add_argument('--threads', default=32, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=60, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='parallel', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    seq_list = None
    if dataset_name == 'GTOT':
        seq_home = './adv/GTOT'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'VOT2020':
        seq_home = './data/VOT2020'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'LasHeR':
        seq_home = '/media/jiawen/Datasets/Tracking/DATASET/LasHeR/testingset'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'VTUAVST':
        seq_home = '/mnt/6196b16a-836e-45a4-b6f2-641dca0991d0/VTUAV/test/short-term'
        with open(join(join(seq_home, 'VTUAV-ST.txt')), 'r') as f:
            seq_list = f.read().splitlines()
    elif dataset_name == 'VTUAVLT':
        seq_home = '/mnt/6196b16a-836e-45a4-b6f2-641dca0991d0/VTUAV/test/long-term'
        with open(join(seq_home, 'VTUAV-LT.txt'), 'r') as f:
            seq_list = f.read().splitlines()
    else:
        raise ValueError("Error dataset!")

    start = time.time()
    if args.mode == 'parallel':
        lr = 500.0
        sigma = 5.0  # NES perturbation scale.
        N_dirs = 5  # Number of NES sampling directions.
        N_iters = 200

        patch_save_dir = f"./Patch_save/{dataset_name}"
        os.makedirs(patch_save_dir, exist_ok=True)

        # Load the initial patch images.
        patch_v = cv2.imread("./Patch_save/visible.jpg").astype(np.float32)
        patch_i = cv2.imread("./Patch_save/infrared.jpg").astype(np.float32)

        if patch_v is None or patch_i is None:
            raise ValueError("visible.jpg or infrared.jpg was not found.")

        H, W, C = patch_v.shape

        sequence_list = [
            (s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name)
            for s in seq_list
        ]

        for epoch in range(N_iters):
            # Initialize the accumulated gradient maps.
            grad_v = np.zeros_like(patch_v)
            grad_i = np.zeros_like(patch_i)

            loss = 0
            for k in range(N_dirs):
                noise_v = np.random.randn(H, W, C).astype(np.float32)
                noise_i = np.random.randn(H, W, C).astype(np.float32)

                # Evaluate positive and negative perturbations.
                pv_plus = np.clip(patch_v + sigma * noise_v, 0, 255)
                pv_minus = np.clip(patch_v - sigma * noise_v, 0, 255)
                pi_plus = np.clip(patch_i + sigma * noise_i, 0, 255)
                pi_minus = np.clip(patch_i - sigma * noise_i, 0, 255)

                print(f"[epoch {epoch}] NES sample {k + 1}/{N_dirs} IoU+")
                process_got_with_patch(pv_plus, pi_plus)
                multiprocessing.set_start_method('spawn', force=True)
                with multiprocessing.Pool(processes=args.threads) as pool:
                    iou_plus_list = pool.starmap(run_sequence, sequence_list)
                L_plus = sum(iou_plus_list) / len(iou_plus_list)
                print("L_plus:", L_plus)

                print(f"[epoch {epoch}] NES sample {k + 1}/{N_dirs} IoU-")
                process_got_with_patch(pv_minus, pi_minus)
                multiprocessing.set_start_method('spawn', force=True)
                with multiprocessing.Pool(processes=args.threads) as pool:
                    iou_minus_list = pool.starmap(run_sequence, sequence_list)
                L_minus = sum(iou_minus_list) / len(iou_minus_list)
                print("L_minus:", L_minus)

                print("delta:", L_plus - L_minus)
                loss += (L_plus + L_minus) / 2

                # Accumulate the gradient contribution from this direction.
                gk_v = -(L_plus - L_minus) / (2 * sigma) * noise_v
                gk_i = -(L_plus - L_minus) / (2 * sigma) * noise_i

                grad_v += gk_v
                grad_i += gk_i

                print(f"Epoch {epoch + 1}/{N_iters} - NES {k + 1}/{N_dirs}: iou_mean={(L_plus + L_minus) / 2}")

            grad_v /= N_dirs
            grad_i /= N_dirs

            # Normalize the gradients before updating the patches.
            grad_v = grad_v / (np.abs(grad_v).mean() + 1e-8)
            grad_i = grad_i / (np.abs(grad_i).mean() + 1e-8)
            grad_i = gray_grad(grad_i)
            loss = loss / N_dirs

            patch_v = np.clip(patch_v + lr * grad_v * loss, 0, 255)
            patch_i = np.clip(patch_i + lr * grad_i * loss, 0, 255)

            print(f"Epoch {epoch + 1}/{N_iters}: grad_v mean={grad_v.mean():.6f}, loss={loss}")

            # Save the updated patches for the current epoch.
            cv2.imwrite(os.path.join(patch_save_dir, f"patch_epoch_{epoch:03d}_v.jpg"), patch_v.astype(np.uint8))
            cv2.imwrite(os.path.join(patch_save_dir, f"patch_epoch_{epoch:03d}_i.jpg"), patch_i.astype(np.uint8))

        process_got_with_patch(patch_v, patch_i)
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            iou_list = pool.starmap(run_sequence, sequence_list)
        mean_iou_all = sum(iou_list) / len(iou_list)
        print("All sequences mean IOU:", mean_iou_all)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [
            (s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name)
            for s in seq_list
        ]
        for seqlist in sequence_list:
            run_sequence(*seqlist)

    print(f"Totally cost {time.time() - start} seconds!")
