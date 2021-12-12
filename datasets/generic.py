import torch
import numpy as np
from torch.utils.data import Dataset
import MinkowskiEngine as ME


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        raise ValueError("Can not convert to torch tensor {}".format(x))


class Batch:
    def __init__(self, batch):
        pc_0, pc_1, coords0, coords1, feats0, feats1, mask, flow, dir_name = list(zip(*batch))

        pc_0_batch = torch.cat(pc_0, 0)
        pc_1_batch = torch.cat(pc_1, 0)
        mask_batch = torch.cat(mask, 0)
        flow_batch = torch.cat(flow, 0)

        coords0_batch, feats0_batch = ME.utils.sparse_collate(coords=coords0, feats=feats0)
        coords1_batch, feats1_batch = ME.utils.sparse_collate(coords=coords1, feats=feats1)

        self.dir_name = dir_name

        self.data = {
            'pc0': pc_0_batch,
            'pc1': pc_1_batch,
            'mask': mask_batch,
            'flow': flow_batch,
            's_C': coords0_batch,
            's_F': feats0_batch,
            't_C': coords1_batch,
            't_F': feats1_batch,
        }

    def __getitem__(self, item):
        """
        Get 'sequence', 'ground_thruth' or 'dir_name' from the batch.

        Args:
            item: str. Accept two keys 'sequence', 'ground_truth' or 'dir_name'.

        Returns:
            item='sequence': returns a list [pc1, pc2] of point clouds between
            which to estimate scene flow. pc1 has size B x n x 3 and pc2 has
            size B x m x 3.

            item='ground_truth': returns a list [mask, flow]. mask has size
            B x n x 1 and flow has size B x n x 3. flow is the ground truth
            scene flow between pc1 and pc2. flow is the ground truth scene
            flow. mask is binary with zeros indicating where the flow is not
            valid or occluded.

            item='dir_name': returns a list [str]. dir_name is the directory
            name of the point cloud data. It can be used for some debug purposes.

        """
        if item == 'dir_name':
            return self.dir_name

        return self.data[item]

    def to(self, *args, **kwargs):
        for key in self.data.keys():
            self.data[key] = self.data[key].to(*args, **kwargs)

        return self

    def pin_memory(self):
        for key in self.data.keys():
            self.data[key] = self.data[key].pin_memory()

        return self


class SceneFlowDataset(Dataset):
    def __init__(self, nb_points, voxel_size=0.05):
        """
        Abstract constructor for scene flow datasets.

        Each item of the dataset is returned in a dictionary with two keys:
            (key = 'sequence', value=list(torch.Tensor, torch.Tensor)):
            list [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size 1 x n x 3 and pc2 has size 1 x m x 3.

            (key = 'ground_truth', value = list(torch.Tensor, torch.Tensor)):
            list [mask, flow]. mask has size 1 x n x 1 and pc1 has size
            1 x n x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not
            valid/occluded.

            (key = 'dir_name', value = list(str)):
            directory names of training data.


        Args:
            nb_points: int. Maximum number of points in point clouds.
            voxel_size: float. Voxel size when do voxelization.
        """

        super(SceneFlowDataset, self).__init__()
        self.nb_points = nb_points
        self.voxel_size = voxel_size

    def __getitem__(self, idx):
        np_sequence, np_ground_truth, dir_name = self.subsample_points(*self.load_sequence(idx))
        sequence, ground_truth = self.to_torch(np_sequence, np_ground_truth)
        pc_0 = sequence[0]  # [1, n, 3]
        pc_1 = sequence[1]  # [1, n, 3]

        mask = ground_truth[0]  # [1, n, 1]
        flow = ground_truth[1]  # [1, n, 3]

        # sel0: unique_idx
        _, sel0 = ME.utils.sparse_quantize(np.ascontiguousarray(pc_0[0]) / self.voxel_size, return_index=True)

        # sel1: unique_idx
        _, sel1 = ME.utils.sparse_quantize(np.ascontiguousarray(pc_1[0]) / self.voxel_size, return_index=True)

        voxelized_pc_0 = pc_0[0, sel0, :]  # [m1, 3]
        voxelized_pc_1 = pc_1[0, sel1, :]  # [m2, 3]

        # Get sparse indices
        coords0 = np.floor(voxelized_pc_0 / self.voxel_size)  # [m1, 3]
        coords1 = np.floor(voxelized_pc_1 / self.voxel_size)  # [m2, 3]

        feats0 = voxelized_pc_0  # [m1, 3]
        feats1 = voxelized_pc_1  # [m2, 3]

        return pc_0, pc_1, coords0, coords1, feats0, feats1, mask, flow, dir_name

    def to_torch(self, sequence, ground_truth):
        """
        Convert numpy array and torch.Tensor.

        Parameters
        ----------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size n x 3 and pc2 has size m x 3.
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size n x 1 and pc1 has size n x 3. 
            flow is the ground truth scene flow between pc1 and pc2. mask is 
            binary with zeros indicating where the flow is not valid/occluded.
        
        Returns
        -------
        sequence : list(torch.Tensor, torch.Tensor)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size 1 x n x 3 and pc2 has size 1 x m x 3.
            
        ground_truth : list(torch.Tensor, torch.Tensor)
            List [mask, flow]. mask has size 1 x n x 1 and pc1 has size 
            1 x n x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not 
            valid/occluded.

        """

        sequence = [torch.unsqueeze(torch.from_numpy(s), 0).float() for s in sequence]
        ground_truth = [
            torch.unsqueeze(torch.from_numpy(gt), 0).float() for gt in ground_truth
        ]

        return sequence, ground_truth

    def subsample_points(self, sequence, ground_truth, dir_name):
        """
        Subsample point clouds randomly.

        Parameters
        ----------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size 1 x N x 3 and pc2 has size 1 x M x 3.
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size 1 x N x 1 and pc1 has size 
            1 x N x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not 
            valid/occluded.

        dir_name: list(str)

        Returns
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size 1 x n x 3 and pc2 has size 1 x m x 3. The n 
            points are chosen randomly among the N available ones. The m points
            are chosen randomly among the M available ones. If N, M >= 
            self.nb_point then n, m = self.nb_points. If N, M < 
            self.nb_point then n, m = N, M. 
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size 1 x n x 1 and pc1 has size 
            1 x n x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not 
            valid/occluded.

        """

        # Choose points in first scan
        ind1 = np.random.permutation(sequence[0].shape[0])[: self.nb_points]
        sequence[0] = sequence[0][ind1]
        ground_truth = [g[ind1] for g in ground_truth]

        # Choose point in second scan
        ind2 = np.random.permutation(sequence[1].shape[0])[: self.nb_points]
        sequence[1] = sequence[1][ind2]

        return sequence, ground_truth, dir_name

    def load_sequence(self, idx):
        """
        Abstract function to be implemented to load a sequence of point clouds.

        Args:
            idx: int. Index of the sequence to load.

        Returns:
            sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size N x 3 and pc2 has size M x 3.

            ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size N x 1 and pc1 has size N x 3.
            flow is the ground truth scene flow between pc1 and pc2. mask is
            binary with zeros indicating where the flow is not valid/occluded.

            dir_name: lsit(str)

        """
        raise NotImplementedError
