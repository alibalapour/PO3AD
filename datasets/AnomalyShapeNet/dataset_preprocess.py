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


class Dataset:
    def __init__(self, cfg):
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

        # Initializes the training data loader with the specified parameters and custom collate function. Note that collate_fn is a custom function (self.trainMerge) to merge and preprocess data for each batch.
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge,
                                            num_workers=self.dataset_workers,
                                            shuffle=True, sampler=None,
                                            drop_last=True, pin_memory=False,
                                            worker_init_fn=self._worker_init_fn_)

    def testLoader(self):
        # Creates test dataset indecies.
        test_set = list(range(len(self.test_file_list)))

        # Initializes the test data loader with the specified parameters and custom collate function. Note that collate_fn is a custom function (self.testMerge) to merge and preprocess data for each batch.
        self.test_data_loader = DataLoader(test_set, batch_size=1, collate_fn=self.testMerge,
                                           num_workers=self.dataset_workers,
                                           shuffle=False, sampler=None,
                                           drop_last=False, pin_memory=False,
                                           worker_init_fn=self._worker_init_fn_)

    def generate_pseudo_anomaly(self, points, normals, center, distance_to_move=0.08):
        
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

    def trainMerge(self, id):
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
        for i, idx in enumerate(id):
            fn_path = self.train_file_list[idx]  # get path
            file_name.append(self.train_file_list[idx])

            # #####Load data
            obj = o3d.io.read_triangle_mesh(fn_path)
            obj.compute_vertex_normals()
            coord = np.asarray(obj.vertices)
            vertex_normals = np.asarray(obj.vertex_normals)
            mask = np.ones(coord.shape[0]) * -1

            # ####Data aug
            Point_dict = {'coord': coord,
                          'normal': vertex_normals, 'mask': mask}
            Point_dict, centers = self.train_aug_compose(Point_dict)

            # ####Trans to numpy
            xyz = Point_dict['coord'].astype(np.float32)
            normal = Point_dict['normal'].astype(np.float32)
            mask = Point_dict['mask'].astype(np.int32)

            # ensure that the mask label is between 0 to self.mask_num-1
            mask[mask == (self.mask_num + 1)] = self.mask_num - 1

            xyz_original.append(torch.from_numpy(xyz))

            # ####Generate pseudo anomaly

            # Select random regions to create pseudo anomalies by shifting points within those regions.
            num_shift = 1
            mask_range = np.arange(0, self.mask_num // 2)

            # Randomly selects regions from the first half of the mask indices to create pseudo anomalies.
            shift_index = np.random.choice(
                mask_range, num_shift, replace=False)

            # Updates the mask to mark the selected regions for shifting with -1.
            mask[np.isin(mask, shift_index)] = -1

            # Generate pseudo anomaly by shifting the points in the selected mask regions
            shift_xyz = xyz[mask == -1].copy()
            shift_normal = normal[mask == -1].copy()
            shifted_xyz = self.generate_pseudo_anomaly(
                shift_xyz, shift_normal, centers[shift_index[0]], distance_to_move=np.random.uniform(0.06, 0.12))

            new_xyz = xyz.copy()

            new_xyz[mask == -1] = shifted_xyz

            # Calculate the ground truth offset between the original and shifted point clouds
            gt_offset = new_xyz - xyz
            gt_offset_list.append(torch.from_numpy(gt_offset))

            xyz_shifted.append(torch.from_numpy(new_xyz))

            # ####Voxelization
            quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(new_xyz, new_xyz,
                                                                                         quantization_size=self.voxel_size,
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
