import plotly.offline as po
import plotly.graph_objects as go
import os
import sys
import time
import random
import torch
import numpy as np
import open3d as o3d
import torch.optim as optim
from math import cos, pi
from tensorboardX import SummaryWriter

import tools.log as log
from config.config_eval import get_parser
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score

import os
import glob
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def load_checkpoint(model, pretrain_file, gpu=0):
    map_location = {'cuda:0': 'cuda:{}'.format(gpu)} if gpu > 0 else None
    checkpoint = torch.load(pretrain_file, map_location=map_location)
    model_dict = checkpoint['model']
    for k, v in model_dict.items():
        if 'module.' in k:
            model_dict = {k[len('module.'):]: v for k, v in model_dict.items()}
        break
    model.load_state_dict(model_dict, strict=False)


def save_anomalies_visualization(batch, pred_mask, gt_mask=None, output_dir="visualizations", file_prefix="sample", counter=0):
    """
    Save point cloud anomalies visualization to a file.

    Args:
        batch (dict): Batch data containing the original point cloud.
        pred_mask (numpy.ndarray): Predicted anomaly scores for each point.
        gt_mask (numpy.ndarray, optional): Ground truth anomaly mask for each point.
        output_dir (str): Directory to save the visualizations.
        file_prefix (str): Prefix for the output file names.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the original point cloud from the batch
    points = batch['xyz_original'].numpy()

    # Normalize predicted mask for visualization
    pred_mask_normalized = (pred_mask - np.min(pred_mask)) / \
        (np.max(pred_mask) - np.min(pred_mask))

    # Define a custom colormap with three colors: light grey, yellow, and red
    clist = ['lightgrey', 'yellow', 'red']
    newcmp = LinearSegmentedColormap.from_list('chaos', clist)

    # Map the normalized scores to RGB colors using the custom colormap
    colors = plt.get_cmap(newcmp)(pred_mask_normalized)[:, :3]

    # Create a point cloud object and assign the XYZ coordinates and colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save the predicted anomalies visualization
    file_prefix += "_" + str(counter)
    pred_file_path = os.path.join(output_dir, f"{file_prefix}_predicted.ply")
    o3d.io.write_point_cloud(pred_file_path, pcd)
    print(f"Saved predicted anomalies visualization to {pred_file_path}")

    # If ground truth mask is provided, save it as well
    if gt_mask is not None:

        # Assign colors based on ground truth mask
        gt_colors = np.zeros((points.shape[0], 3))
        # Define a custom colormap with three colors: light grey, yellow, and red
        clist = ['lightgrey', 'yellow', 'red']
        newcmp = LinearSegmentedColormap.from_list('chaos', clist)

        # Map the normalized scores to RGB colors using the custom colormap
        colors = plt.get_cmap(newcmp)(gt_mask)[:, :3]
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(points)
        gt_pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save the ground truth anomalies visualization
        gt_file_path = os.path.join(
            output_dir, f"{file_prefix}_ground_truth.ply")
        o3d.io.write_point_cloud(gt_file_path, gt_pcd)
        print(f"Saved ground truth anomalies visualization to {gt_file_path}")


def save_pc_plotly_html(points_xyz, scores, out_html):
    # points_xyz: (N,3) numpy
    # scores: (N,) numpy, any scale
    s = scores.astype(float)
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)  # [0,1]

    # map to RGB (lightgray->yellow->red)
    # lightgray (0.83), yellow, red
    # simple lerp piecewise: 0-0.5 go gray->yellow, 0.5-1 yellow->red
    def lerp(a, b, t): return a*(1-t)+b*t
    N = points_xyz.shape[0]
    rgb = np.zeros((N, 3))
    for i, v in enumerate(s):
        if v < 0.5:
            t = v/0.5
            rgb[i] = [lerp(0.83, 1.0, t), lerp(0.83, 1.0, t),
                      lerp(0.83, 0.0, t)]  # gray→yellow
        else:
            t = (v-0.5)/0.5
            rgb[i] = [1.0, lerp(1.0, 0.0, t), 0.0]  # yellow→red

    fig = go.Figure(data=[go.Scatter3d(
        x=points_xyz[:, 0], y=points_xyz[:, 1], z=points_xyz[:, 2],
        mode='markers',
        marker=dict(size=2, opacity=0.9, color=['rgb({},{},{})'.format(
            int(255*r), int(255*g), int(255*b)) for r, g, b in rgb])
    )])
    fig.update_layout(scene=dict(aspectmode="data"),
                      margin=dict(l=0, r=0, t=0, b=0))
    po.plot(fig, filename=out_html, auto_open=False)


def save_anomalies_visualization_html(batch, pred_mask, gt_mask=None, output_dir="visualizations", file_prefix="sample", counter=0):

    os.makedirs(output_dir, exist_ok=True)
    points = batch['xyz_original'].numpy()
    tag = f"{file_prefix}_{counter}"

    save_pc_plotly_html(points, pred_mask, os.path.join(
        output_dir, f"{tag}_predicted.html"))
    if gt_mask is not None:
        save_pc_plotly_html(points, gt_mask.astype(float), os.path.join(
            output_dir, f"{tag}_ground_truth.html"))


def eval(cfgs):
    global cfg
    cfg = cfgs
    from network.PO3AD import PONet as net
    from network.PO3AD import eval_fn
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    model = net(cfg.in_channels, cfg.out_channels)
    model = model.cuda()
    load_checkpoint(model, cfg.logpath + cfg.checkpoint_name)

    if cfg.dataset == 'AnomalyShapeNet':
        from datasets.AnomalyShapeNet.dataset_preprocess import Dataset
        gt_mask_path = f'datasets/AnomalyShapeNet/dataset/pcd/{cfg.category}/GT/'
        tag = 'positive'
    elif cfg.dataset == 'Real3D':
        from datasets.Real3D.dataset_preprocess import Dataset
        gt_mask_path = f'datasets/Real3D/Real3D-AD-PCD/{cfg.category}/gt/'
        tag = 'good'
    else:
        print('do not support this dataset at present')

    dataset = Dataset(cfg)
    dataset.testLoader()
    print(f'Test samples: {len(dataset.test_file_list)}')

    model.eval()
    label_score = []
    gt_masks = []
    pred_masks = []
    for i, batch in enumerate(dataset.test_data_loader):

        # clean up GPU memory
        torch.cuda.empty_cache()

        # creating sample name
        sample_name = batch['fn'][0].split('/')[-1].split('.')[0]

        # preparing gt_mask based on data type (using origninal for non-anomalous samples and loading generated gt for pseuodo-anomalous samples)
        if tag in sample_name:
            gt_masks.append(np.zeros(batch['xyz_original'].shape[0]))
        else:
            if cfg.dataset == 'AnomalyShapeNet':
                gt_mask = np.loadtxt(
                    gt_mask_path + sample_name + '.txt', delimiter=',')[:, 3:].squeeze(1)
            elif cfg.dataset == 'Real3D':
                gt_mask = np.loadtxt(
                    gt_mask_path + sample_name + '.txt')[:, 3:].squeeze(1)
            gt_masks.append(gt_mask)

        score, pred_mask = eval_fn(batch, model)
        pred_mask = pred_mask.detach().cpu().abs().sum(dim=-1).numpy()
        pred_masks.append(pred_mask)
        label_score += list(zip(batch['labels'].numpy().tolist(),
                            [score.item()]))
        # print("*"*20), print(f'[{i}/{len(dataset.test_file_list)}] {batch["fn"][0]} | label: {batch["labels"].numpy()[0]} | score: {score.item()}, | pred mask: {pred_mask}'), print("*"*20)
        # save_anomalies_visualization(
        #     batch, pred_mask, gt_mask, output_dir="results", file_prefix="test_sample", counter=i)
        # save_anomalies_visualization_html(
        #     batch, pred_mask, gt_mask, output_dir="results_html", file_prefix="test_sample", counter=i)

    labels, scores = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    auc_roc = roc_auc_score(labels, scores)
    auc_pr = average_precision_score(labels, scores)
    point_pred = np.concatenate(pred_masks, axis=0)
    point_pred = (point_pred - np.min(point_pred)) / \
        (np.max(point_pred) - np.min(point_pred))
    point_auc_roc = roc_auc_score(np.concatenate(gt_masks, axis=0), point_pred)
    point_auc_pr = average_precision_score(
        np.concatenate(gt_masks, axis=0), point_pred)
    print(
        f'object AUC-ROC: {auc_roc}, point AUC-ROC: {point_auc_roc}, object AUCP-PR: {auc_pr}, point AUCP-PR: {point_auc_pr}')


if __name__ == '__main__':
    cfg = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    eval(cfg)
