import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from models.scene_flow import SCTN
from torch.utils.data import DataLoader
from datasets.generic import Batch


def compute_epe(est_flow, batch):
    """
    Compute EPE, accuracy and number of outliers.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    EPE3D : float
        End point error.
    acc3d_strict : float
        Strict accuracy.
    acc3d_relax : float
        Relax accuracy.
    outlier : float
        Percentage of outliers.

    """

    # Extract occlusion mask
    mask = batch["mask"].cpu().numpy()[..., 0]

    # Flow
    sf_gt = batch["flow"].cpu().numpy()[mask > 0]
    sf_pred = est_flow.cpu().numpy()[mask > 0]

    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)
    EPE3D = l2_norm.mean()

    sf_norm = np.linalg.norm(sf_gt, axis=-1)
    relative_err = l2_norm / (sf_norm + 1e-4)
    acc3d_strict = (
        (np.logical_or(l2_norm < 0.05, relative_err < 0.05)).astype(np.float).mean()
    )
    acc3d_relax = (
        (np.logical_or(l2_norm < 0.1, relative_err < 0.1)).astype(np.float).mean()
    )
    outlier = (np.logical_or(l2_norm > 0.3, relative_err > 0.1)).astype(np.float).mean()

    return EPE3D, acc3d_strict, acc3d_relax, outlier


def eval_model(scene_flow, testloader, ckpt_iter):
    """

    Args:
        scene_flow: SCTN model.
        testloader: DataLoader for the test dataset.
        ckpt_iter: int. Iteration number of the loaded checkpoint.

    Returns:
        None.
    """

    # Init.
    running_epe = 0
    running_outlier = 0
    running_acc3d_relax = 0
    running_acc3d_strict = 0

    scene_flow = scene_flow.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for it, batch in enumerate(tqdm(testloader)):

        # Send data to GPU
        batch = batch.to(device, non_blocking=True)

        # Estimate flow
        with torch.no_grad():
            est_flow = scene_flow(batch["pc0"], batch["pc1"], batch["s_C"], batch["s_F"], batch["t_C"], batch["t_F"])

        # Perf. metrics
        EPE3D, acc3d_strict, acc3d_relax, outlier = compute_epe(est_flow, batch)
        running_epe += EPE3D
        running_outlier += outlier
        running_acc3d_relax += acc3d_relax
        running_acc3d_strict += acc3d_strict

    mean_epe = running_epe / (it + 1)
    mean_outlier = running_outlier / (it + 1)
    mean_acc3d_relax = running_acc3d_relax / (it + 1)
    mean_acc3d_strict = running_acc3d_strict / (it + 1)

    print(f""
          f"ckpt_iter:{ckpt_iter}\n"
          f"EPE: {mean_epe}\n"
          f"Outlier: {mean_outlier}\n"
          f"ACC3DR: {mean_acc3d_relax}\n"
          f"ACC3DS: {mean_acc3d_strict}\n"
          f"Dataset len: {len(testloader)}"
          f"")


def run_test(dataset_name, max_points, path2ckpt, voxel_size):
    """
    Entry point of the test script.

    Args:
        dataset_name: str. Dataset on which to evaluate. Either HPLFlowNet_kitti or HPLFlowNet_FT3D.
        max_points: int. Number of points in point clouds.
        path2ckpt: str. Path to saved model.
        voxel_size: Voxel size when doing voxelization.

    Returns:
        Print out info including: EPE, Outlier, ACC3DR, ACC3DS and the length of the test dataset.
    """

    # Path to current file
    pathroot = os.path.dirname(__file__)

    # Select dataset
    if dataset_name.split("_")[0].lower() == "HPLFlowNet".lower():

        # HPLFlowNet version of the datasets
        path2data = os.path.join(pathroot, "data", "HPLFlowNet")

        # KITTI
        if dataset_name.split("_")[1].lower() == "kitti".lower():
            path2data = os.path.join(path2data, "KITTI_processed_occ_final")
            from datasets.kitti_hplflownet import Kitti
            dataset = Kitti(root_dir=path2data, nb_points=max_points, voxel_size=voxel_size)

        # FlyingThing3D
        elif dataset_name.split("_")[1].lower() == "ft3d".lower():
            path2data = os.path.join(path2data, "FlyingThings3D_subset_processed_35m")
            from datasets.flyingthings3d_hplflownet import FT3D
            dataset = FT3D(root_dir=path2data, nb_points=max_points, mode="test", voxel_size=voxel_size)

        else:
            raise ValueError("Unknown dataset " + dataset_name)
    else:
        raise ValueError("Unknown dataset " + dataset_name)
    # print("\n\nDataset: " + path2data + " " + mode)

    # Dataloader
    testloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        num_workers=6,
        collate_fn=Batch,
        drop_last=False,
    )

    # Load SCTN model
    scene_flow = SCTN(nb_iter=None, voxel_size=voxel_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scene_flow = scene_flow.to(device, non_blocking=True)
    file = torch.load(path2ckpt)
    scene_flow.nb_iter = file["nb_iter"]
    scene_flow.load_state_dict(file["model"])
    scene_flow = scene_flow.eval()

    ckpt_iter = os.path.basename(path2ckpt)[:-4][6:]
    eval_model(scene_flow, testloader, ckpt_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SCTN.")

    parser.add_argument(
        "--dataset",
        type=str,
        default="HPLFlowNet_FT3D",
        help="Dataset. Either HPLFlowNet_FT3D or HPLFlowNet_kitti",
    )

    parser.add_argument(
        "--nb_points",
        type=int,
        default=8192,
        help="Maximum number of points in point cloud.",
    )

    parser.add_argument(
        "--path2ckpt",
        type=str,
        default='pre-trained/model-59.tar',
        help="Path to saved checkpoint.",
    )

    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.07,
        help="voxel size",
    )

    args = parser.parse_args()

    # Launch test
    run_test(args.dataset, args.nb_points, args.path2ckpt, args.voxel_size)
