import multiprocessing as mp
import math
import glob
import torch
import random
import numpy as np
import open3d as o3d
import scipy.ndimage
import scipy.interpolate
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
import datasets.AnomalyShapeNet.transform as aug_transform
import os
import re

import numpy as np
from dataclasses import dataclass
import open3d as o3d
import MinkowskiEngine as ME
import torch


from eval import save_pc_plotly_html
DEBUG = False


# To inject params into dataloader workers

manager = mp.Manager()
param_queue = manager.Queue()


# Contribution: Writing dataloader collate function as a closure to capture shared config that can be modified externally through training loop, without any signi


@dataclass
class SmartAnomaly_Cfg:
    # size & strength
    R: float = None            # support radius; None -> 0.2 * object diameter
    beta: float = 0.08                # magnitude (your distance_to_move)
    alpha: int = None          # +1 bulge, -1 cavity, None -> random w.p. p_bulge
    p_bulge: float = 0.5

    # falloff kernel
    kernel: str = "cosine"            # {"cosine","gaussian","poly","hard"}
    q: float = 2.0
    sigma: float = 0.35               # as fraction of R for gaussian

    # anisotropy (ellipsoid radii along local frame axes)
    radii: tuple = (1.0, 1.0, 1.0)    # (ru, rv, rn); 1,1,1 == sphere

    # displacement direction
    # {"normal_point","normal_mean","tangent_u","tangent_v"}
    dir_mode: str = "normal_point"

    # extras
    # 0..1 probability of removing center points (holes)
    carve_strength: float = 0.0
    smooth_steps: int = 0             # Laplacian smoothing steps inside support
    smooth_lambda: float = 0.15
    seed: int = None


def make_collate(dataset_object, param_queue):
    import queue as _queue

    def trainMerge_with_anomlay_cfg(id_list):        # Snapshot once per batch

        # w = torch.utils.data.get_worker_info()
        # worker_id = w.id if w is not None else -1

        # Getting parameters of pseudo anomlay synthesis from a queue during training
        default_beta = 0.08
        queue_timeout = 1.0
        try:
            params = param_queue.get(timeout=queue_timeout)
        except _queue.Empty:
            params = {'beta': default_beta}
        beta = float(params.get('beta', default_beta))

        file_name = []
        xyz_voxel = []
        feat_voxel = []
        xyz_original = []
        xyz_shifted = []
        v2p_index_batch = []
        total_voxel_num = 0
        batch_count = [0]
        total_point_num = 0
        gt_offset_list = []
        for i, idx in enumerate(id_list):
            fn_path = dataset_object.train_file_list[idx]  # get path
            file_name.append(dataset_object.train_file_list[idx])

            # #####Load data
            obj = o3d.io.read_triangle_mesh(fn_path)
            obj.compute_vertex_normals()
            coord = np.asarray(obj.vertices)
            vertex_normals = np.asarray(obj.vertex_normals)
            mask = np.ones(coord.shape[0]) * -1

            # ####Data aug
            Point_dict = {'coord': coord,
                          'normal': vertex_normals, 'mask': mask}
            Point_dict, centers = dataset_object.train_aug_compose(Point_dict)

            # ####Trans to numpy
            xyz = Point_dict['coord'].astype(np.float32)
            normal = Point_dict['normal'].astype(np.float32)
            mask = Point_dict['mask'].astype(np.int32)

            # ensure that the mask label is between 0 to dataset_object.mask_num-1
            mask[mask == (dataset_object.mask_num + 1)
                 ] = dataset_object.mask_num - 1

            xyz_original.append(torch.from_numpy(xyz))

            # ####Generate pseudo anomaly

            # Select random regions to create pseudo anomalies by shifting points within those regions.
            num_shift = 1
            mask_range = np.arange(0, dataset_object.mask_num // 2)

            # Randomly selects regions from the first half of the mask indices to create pseudo anomalies.
            shift_index = np.random.choice(
                mask_range, num_shift, replace=False)

            # Updates the mask to mark the selected regions for shifting with -1.
            mask[np.isin(mask, shift_index)] = -1

            # Generate pseudo anomaly by shifting the points in the selected mask regions
            shift_xyz = xyz[mask == -1].copy()
            shift_normal = normal[mask == -1].copy()
            # shifted_xyz = dataset_object.generate_pseudo_anomaly(
            #     shift_xyz, shift_normal, centers[shift_index[0]], distance_to_move=np.random.uniform(0.06, 0.12))

            anomlay_cfg, R_frac = dataset_object.sample_anomlay_cfg(np.random.default_rng())
            
            if dataset_object.global_cfg.smart_anomaly:
                shifted_xyz = dataset_object.generate_pseudo_anomaly(
                    shift_xyz, shift_normal, centers[shift_index[0]], distance_to_move=beta,
                    anomlay_cfg=anomlay_cfg)
            else:
                shifted_xyz = dataset_object.generate_pseudo_anomaly_original(
                    shift_xyz, shift_normal, centers[shift_index[0]], distance_to_move=beta)

            new_xyz = xyz.copy()

            new_xyz[mask == -1] = shifted_xyz

            # Calculate the ground truth offset between the original and shifted point clouds
            gt_offset = new_xyz - xyz
            gt_offset_list.append(torch.from_numpy(gt_offset))

            if DEBUG:
                save_pc_plotly_html(new_xyz, gt_offset.sum(
                    axis=-1), f'debug/shifted_{i}.html')
                print(
                    "*"*20, *f"Saved shifted points visualization to debug/shifted_{i}.html")

            xyz_shifted.append(torch.from_numpy(new_xyz))

            # ####Voxelization
            quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(new_xyz, new_xyz,
                                                                                         quantization_size=dataset_object.voxel_size,
                                                                                         return_index=True,
                                                                                         return_inverse=True)

            # Calculate voxel to point mapping
            v2p_index = inverse_index + total_voxel_num
            total_voxel_num = total_voxel_num + index.shape[0]

            total_point_num += inverse_index.shape[0]
            batch_count.append(total_point_num)

            # -------------------------------Batch -------------------------
            #  merge the scene to the batch
            xyz_voxel.append(quantized_coords)
            feat_voxel.append(feats_all)
            v2p_index_batch.append(v2p_index)
        # ####numpy to torch

        # Collates the voxelized coordinates and features into a batch using MinkowskiEngine's sparse_collate function.
        xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(
            xyz_voxel, feat_voxel)

        # Concatenates the original point clouds from all samples in the batch into a single tensor.
        xyz_original = torch.cat(xyz_original, 0).to(torch.float32)

        # Concatenates the shifted point clouds from all samples in the batch into a single tensor.
        xyz_shifted = torch.cat(xyz_shifted, 0).to(torch.float32)

        # Concatenates the voxel-to-point index mappings from all samples in the batch into a single tensor.
        v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)

        # Converts the batch count list to a PyTorch tensor.
        batch_count = torch.from_numpy(np.array(batch_count))

        # Concatenates the ground truth offsets from all samples in the batch into a single tensor.
        batch_offset = torch.cat(gt_offset_list, 0).to(torch.float32)

        return {'xyz_voxel': xyz_voxel_batch, 'feat_voxel': feat_voxel_batch, 'xyz_original': xyz_original,
                'fn': file_name, 'v2p_index': v2p_index_batch, 'xyz_shifted': xyz_shifted, 'batch_count': batch_count, 'batch_offset': batch_offset}
    return trainMerge_with_anomlay_cfg


class Dataset:
    def __init__(self, cfg):
        self.global_cfg = cfg
        self.batch_size = cfg.batch_size
        self.dataset_workers = cfg.num_works
        self.data_repeat = cfg.data_repeat
        self.voxel_size = cfg.voxel_size
        self.mask_num = cfg.mask_num

        self.category = cfg.category
        self.category_list = os.listdir(
            'datasets/AnomalyShapeNet/dataset/pcd/')
        assert self.category in self.category_list

        data_list = glob.glob(
            "datasets/AnomalyShapeNet/dataset/obj/{}/*.obj".format(self.category))

        # Prepares a list of training files by filtering and repeating relevant files. Note that for training it is using "template" files: e.g. bag0_template0.obj
        is_train = re.compile(r'template')
        self.train_file_list = list(filter(is_train.search, data_list))
        self.train_file_list.sort()
        self.train_file_list = self.train_file_list * self.data_repeat

        # Collects all .pcd files in the test directory for evaluation.
        self.test_file_list = glob.glob(
            "datasets/AnomalyShapeNet/dataset/pcd/{}/test/*.pcd".format(self.category))
        self.test_file_list.sort()

        self.NormalizeCoord = aug_transform.NormalizeCoord()
        self.CenterShift = aug_transform.CenterShift(apply_z=True)
        self.RandomRotate_z = aug_transform.RandomRotate(
            angle=[-1, 1], axis="z", center=[0, 0, 0], p=1.0)
        self.RandomRotate_y = aug_transform.RandomRotate(
            angle=[-1, 1], axis="y", p=1.0)
        self.RandomRotate_x = aug_transform.RandomRotate(
            angle=[-1, 1], axis="x", p=1.0)
        self.SphereCropMask = aug_transform.SphereCropMask(
            part_num=self.mask_num)

        self.train_aug_compose = aug_transform.Compose([self.CenterShift, self.RandomRotate_z, self.RandomRotate_y, self.RandomRotate_x,
                                                        self.NormalizeCoord, self.SphereCropMask])

        self.test_aug_compose = aug_transform.Compose(
            [self.CenterShift, self.NormalizeCoord])

    def _worker_init_fn_(self, worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)
        random.seed(np_seed)

    def trainLoader(self):
        # Creates training dataset indecies.
        train_set = list(range(len(self.train_file_list)))

        self.train_data_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            collate_fn=make_collate(self, param_queue),
            num_workers=self.dataset_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=False,
            worker_init_fn=self._worker_init_fn_,
            persistent_workers=True,  # recommended for speed; safe with Manager proxy
            prefetch_factor=1,               # important: one batch prefetched per worker
        )

    def testLoader(self):
        # Creates test dataset indecies.
        test_set = list(range(len(self.test_file_list)))

        # Initializes the test data loader with the specified parameters and custom collate function. Note that collate_fn is a custom function (self.testMerge) to merge and preprocess data for each batch.
        self.test_data_loader = DataLoader(test_set, batch_size=1, collate_fn=self.testMerge,
                                           num_workers=self.dataset_workers,
                                           shuffle=False, sampler=None,
                                           drop_last=False, pin_memory=False,
                                           worker_init_fn=self._worker_init_fn_)


    def generate_pseudo_anomaly_original(self, points, normals, center, distance_to_move=0.08):

        # print(f"distance_to_move: {distance_to_move}")

        # Find distance of each point to the center
        distances_to_center = np.linalg.norm(points - center, axis=1)

        # Find maximum distance of all points to the center
        max_distance = np.max(distances_to_center)

        # Finds a ratio for each point based on its distance to the center. Points closer to the center will have a higher ratio, meaning they will move more.
        movement_ratios = 1 - (distances_to_center / max_distance)

        # Normalizes the movement ratios to be between 0 and 1
        movement_ratios = (movement_ratios - np.min(movement_ratios)) / \
            (np.max(movement_ratios) - np.min(movement_ratios))

        # Randomly assigns a direction (inward or outward) for each point to move
        directions = np.ones(points.shape[0]) * np.random.choice([-1, 1])

        # Calculates the actual movement for each point
        movements = movement_ratios * distance_to_move * directions

        # Moves the points along their normals by the calculated movements
        new_points = points + np.abs(normals) * movements[:, np.newaxis]

        return new_points

    # -------- Helpers --------

    def _kernel(self, t, kind="cosine", q=2.0, sigma=0.35):
        """t is normalized distance; returns falloff in [0,1]."""
        t = np.clip(t, 0.0, None)
        if kind == "cosine":                    # smooth spherical cap
            x = np.clip(t, 0.0, 1.0)
            return 0.5 * (1 + np.cos(np.pi * x))
        if kind == "gaussian":                  # compact-ish, smooth
            return np.exp(-(t**2) / (2 * (sigma**2)))
        if kind == "poly":                      # (1 - t^q)+
            return np.clip(1.0 - t**q, 0.0, 1.0)
        if kind == "hard":                      # hard support
            return (t < 1.0).astype(t.dtype if hasattr(t, "dtype") else np.float32)
        raise ValueError(f"Unknown kernel: {kind}")

    def _local_frame(self, points, center, k=64):
        """PCA frame around center: columns ~ (tangent_u, tangent_v, normal)."""
        d = np.linalg.norm(points - center, axis=1)
        idx = np.argsort(d)[:k]
        Q = points[idx] - points[idx].mean(0)
        C = Q.T @ Q / max(len(idx)-1, 1)
        w, V = np.linalg.eigh(C)
        V = V[:, np.argsort(w)[::-1]]   # sort desc
        return V  # shape (3,3)

    def _beta_sample(self, rng, a, b):
        return rng.beta(a, b)

    def _choose(self, rng, items, probs=None):
        return items[rng.choice(len(items), p=probs)]

    def sample_anomlay_cfg(self, rng: np.random.Generator) -> SmartAnomaly_Cfg:
        # Discrete choices (minimal action set)
        kernel = self._choose(rng, ["cosine", "gaussian", "poly", "hard"], probs=[
                              0.4, 0.4, 0.1, 0.1])
        dir_mode = self._choose(rng, ["normal_point", "normal_mean", "tangent_u", "tangent_v"], probs=[
                                0.925, 0.025, 0.025, 0.025])
        alpha = 1 if rng.random() < 0.5 else -1
        # Smoothing: mostly 0, sometimes 1, rarely 2
        smooth_steps = rng.choice([0, 1, 2], p=[0.7, 0.25, 0.05])

        # Continuous (Beta → range)
        # R: fraction of diameter in ????
        u_R = self._beta_sample(rng, 1, 1)
        R_frac = 0.01 + (0.2 - 0.08) * u_R
        # beta (strength) in ????
        u_B = self._beta_sample(rng, 1, 1)
        beta = 0.05 + (0.10 - 0.02) * u_B
        # gaussian sigma in [0.20, 0.60]; used only if kernel=="gaussian"
        u_S = self._beta_sample(rng, 1, 1)
        sigma = 0.20 + (0.60 - 0.20) * u_S
        # anisotropy e in [0.7, 1.8] → radii (e, 1/e, 1)
        u_E = self._beta_sample(rng, 1, 1)
        e = 0.70 + (1.80 - 0.70) * u_E
        radii = (float(e), float(1.0 / e), 1.0)

        # carve strength in [0, 0.4] (mostly small)
        u_C = self._beta_sample(rng, 1, 12)
        carve_strength = 0.0 + 0.40 * u_C
        # smoothing lambda in [0.10, 0.25]
        smooth_lambda = rng.uniform(0.10, 0.25)

        return SmartAnomaly_Cfg(
            R=None,                    # we’ll set absolute R from diameter below
            beta=float(beta),
            alpha=int(alpha),
            p_bulge=0.5,
            kernel=kernel,
            q=2.0,
            sigma=float(sigma),
            radii=radii,
            dir_mode=dir_mode,
            carve_strength=float(carve_strength),
            smooth_steps=int(smooth_steps),
            smooth_lambda=float(smooth_lambda),
            seed=int(rng.integers(0, 2**31 - 1))
        ), R_frac

    # -------- Smart synthesizer (drop-in) --------

    def generate_pseudo_anomaly(self, points, normals, center, distance_to_move=0.08,
                                anomlay_cfg=None):
        """
        Upgraded version of your function. Pass `anomlay_cfg` to enable smart behavior.
        If `anomlay_cfg` is None, it behaves almost like your original (cosine cap, spherical).
        """
        # --- Defaults to preserve your old behavior ---
        if anomlay_cfg is None:
            anomlay_cfg = SmartAnomaly_Cfg(beta=distance_to_move)

        # TODO
        rng = np.random.default_rng(42)

        P = points.astype(np.float32, copy=False)
        N = normals.astype(np.float32, copy=False)
        c = center.astype(np.float32, copy=False)

        # Normalize normals softly; (your code used abs(normals) which kills direction)
        nrm = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
        N = N / nrm

        # Determine radius
        diam = float(np.linalg.norm(P.max(0) - P.min(0)))
        R = anomlay_cfg.R if anomlay_cfg.R is not None else 0.2 * diam

        # Local PCA frame for anisotropy & tangents
        U = self._local_frame(P, c)  # columns: u, v, (approx) n
        ru, rv, rn = anomlay_cfg.radii

        # Coordinates in local frame and anisotropic distance
        X = (P - c) @ U         # (N,3)
        # Mahalanobis-like norm: t=1 on the ellipsoid surface
        invQ = np.diag([1.0/((ru*R)+1e-12)**2,
                        1.0/((rv*R)+1e-12)**2,
                        1.0/((rn*R)+1e-12)**2])
        t = np.sqrt(np.sum((X @ invQ) * X, axis=1))

        # Falloff weights
        w = self._kernel(t, anomlay_cfg.kernel, anomlay_cfg.q, anomlay_cfg.sigma)

        # Direction field
        if anomlay_cfg.dir_mode == "normal_point":
            D = N
        elif anomlay_cfg.dir_mode == "normal_mean":
            D = np.repeat(U[:, 2][None, :], len(P), axis=0)
        elif anomlay_cfg.dir_mode == "tangent_u":
            D = np.repeat(U[:, 0][None, :], len(P), axis=0)
        elif anomlay_cfg.dir_mode == "tangent_v":
            D = np.repeat(U[:, 1][None, :], len(P), axis=0)
        else:
            raise ValueError(f"Unknown dir_mode: {anomlay_cfg.dir_mode}")

        # Alpha (+1/-1)
        alpha = anomlay_cfg.alpha
        if alpha is None:
            alpha = 1 if rng.random() < anomlay_cfg.p_bulge else -1

        # Magnitude
        beta = anomlay_cfg.beta if anomlay_cfg.beta is not None else distance_to_move
        disp = (alpha * beta * w)[:, None] * D
        new_points = P + disp

        # # Optional carving (hole)
        # keep_mask = np.ones(len(P), dtype=bool)
        # if anomlay_cfg.carve_strength > 0.0:
        #     # remove with prob rising toward center using (1 - t)^2 inside support
        #     pr = anomlay_cfg.carve_strength * np.clip(1.0 - np.clip(t, 0, 1), 0, 1)**2
        #     keep_mask = rng.random(len(P)) > pr
        #     new_points = new_points[keep_mask]

        # # Optional light Laplacian smoothing on the deformed region
        # if anomlay_cfg.smooth_steps > 0:
        #     sub_idx = np.where(w[keep_mask] > 0.01)[0]
        #     if len(sub_idx) >= 8:
        #         sub_pts = new_points[sub_idx]
        #         k = min(16, len(sub_pts))
        #         kdt = cKDTree(sub_pts)
        #         _, nn = kdt.query(sub_pts, k=k)
        #         Xs = sub_pts.copy()
        #         for _ in range(anomlay_cfg.smooth_steps):
        #             nbr_mean = Xs[nn].mean(axis=1)
        #             Xs = Xs + anomlay_cfg.smooth_lambda * (nbr_mean - Xs)
        #         new_points[sub_idx] = Xs

        return new_points

    def testMerge(self, id):
        file_name = []
        xyz_voxel = []
        feat_voxel = []
        xyz_original = []
        v2p_index_batch = []
        labels = []

        total_voxel_num = 0
        total_point_num = 0
        batch_count = [0]
        for i, idx in enumerate(id):
            fn_path = self.test_file_list[idx]  # get path
            file_name.append(self.test_file_list[idx])

            if 'positive' in fn_path:
                pcd = o3d.io.read_point_cloud(fn_path)
                coord = np.asarray(pcd.points)
            else:
                sample_name = fn_path.split('/')[-1].split('.')[0]
                gt_mask_path = f'datasets/AnomalyShapeNet/dataset/pcd/{self.category}/GT/'
                coord = np.loadtxt(gt_mask_path + sample_name +
                                   '.txt', delimiter=',')[:, 0:3]

            # ####Data aug
            Point_dict = {'coord': coord}
            Point_dict = self.test_aug_compose(Point_dict)

            # ####Trans to numpy
            xyz = Point_dict['coord'].astype(np.float32)

            quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(xyz, xyz,
                                                                                         quantization_size=self.voxel_size,
                                                                                         return_index=True,
                                                                                         return_inverse=True)

            v2p_index = inverse_index + total_voxel_num
            total_voxel_num = total_voxel_num + index.shape[0]
            total_point_num += inverse_index.shape[0]
            batch_count.append(total_point_num)

            # -------------------------------Batch -------------------------
            #  merge the scene to the batch
            xyz_voxel.append(quantized_coords)
            feat_voxel.append(feats_all)
            xyz_original.append(torch.from_numpy(xyz))
            # normal_original.append(torch.from_numpy(normal))
            v2p_index_batch.append(v2p_index)
            if 'positive' in fn_path:
                labels.append(0)
            else:
                labels.append(1)

        # ####numpy to torch
        xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(
            xyz_voxel, feat_voxel)
        xyz_original = torch.cat(xyz_original, 0).to(torch.float32)
        v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)
        labels = torch.from_numpy(np.array(labels))
        batch_count = torch.from_numpy(np.array(batch_count))
        return {'xyz_voxel': xyz_voxel_batch, 'feat_voxel': feat_voxel_batch, 'xyz_original': xyz_original,
                'fn': file_name, 'v2p_index': v2p_index_batch, 'labels': labels, 'batch_count': batch_count}
