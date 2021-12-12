from collections import OrderedDict

import os
os.environ["OMP_NUM_THREADS"] = "24"
# os.environ["NCCL_DEBUG"]="INFO"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from models.graph import Graph

import time
import torch
import argparse
from datetime import datetime
from datasets.generic import Batch
from models.scene_flow import SCTN
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist


def compute_epe(est_flow, batch):
    """
    Compute EPE during training.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    epe : torch.Tensor
        Mean EPE for current batch.

    """

    mask = batch["mask"][..., 0]
    true_flow = batch["flow"]
    error = est_flow - true_flow
    error = error[mask > 0]
    epe_per_point = torch.sqrt(torch.sum(torch.pow(error, 2.0), -1))
    epe = epe_per_point.mean()

    return epe


def extract_neighbor_info(signal, graph):
    """
    Extract neighbor info according to graph object.

    Args:
        signal: [b, n, c]. It can be point cloud, feature or scene flow.
        graph: Graph Object.

    Returns:
        (b, n, graph.k_neighbors, c]

    """
    b, n, c = signal.shape
    signal = signal.reshape(b * n, c)
    neighbors_signal = signal[graph.edges]  # [b * n * k_neighbor, c]
    neighbors_signal = neighbors_signal.view(b, n, graph.k_neighbors, c)  # [b, n, k_neighbor, c]
    return neighbors_signal


def compute_fsc_loss(est_flow, feature, graph):
    """
    Compute flow consistent loss by cosine distance
    Args:
        est_flow: estimated flow with shape [b, n, 3]
        feature: feature for point cloud, shape: [b, n, c]
        graph: models.graph.Graph. Graph object for the point cloud.
    Returns:
        loss: shape [b, n]
    """
    feature = feature.detach()

    est_neighbor_flow = extract_neighbor_info(est_flow, graph)  # [b, n, k_neighbor, 3]
    est_diff_flow = est_neighbor_flow - est_flow.unsqueeze(2)  # [b, n, k_neighbor, 3]
    est_local_consistent_loss = torch.norm(est_diff_flow, dim=3)  # [b, n, k_neighbor]

    neighbor_feature = extract_neighbor_info(feature, graph)  # [b, n, k_neighbors, c]
    b, n, k, c = neighbor_feature.shape
    neighbor_feature = neighbor_feature.reshape(b * n, k, c)
    neighbor_feature_norm = neighbor_feature / torch.norm(neighbor_feature, p=2, dim=2, keepdim=True)  # [b * n, k, c]

    center_feature = feature.unsqueeze(2)  # [b, n, 1, c]
    center_feature = center_feature.reshape(b * n, 1, c)  # [b * n, 1, c]
    center_feature_norm = center_feature / torch.norm(center_feature, p=2, dim=2, keepdim=True)  # [b * n, 1, c]

    neighbor_similarity = torch.bmm(center_feature_norm, neighbor_feature_norm.transpose(1, 2))  # [b * n, 1, k]
    neighbor_similarity = neighbor_similarity.reshape(b, n, k)  # [b, n, k]
    tao = 0.45

    neighbor_similarity = torch.div(neighbor_similarity, tao)
    neighbor_similarity = torch.exp(-neighbor_similarity)
    neighbor_similarity = 1 - neighbor_similarity
    neighbor_loss_weight = neighbor_similarity  # [b, n, k]
    weighted_loss = est_local_consistent_loss * neighbor_loss_weight  # [b, n, k]

    fsc_loss = weighted_loss.sum(dim=2) / (k - 1)  # [b, n]
    return fsc_loss


def compute_loss(est_flow, batch, feats, return_fsc=True):
    """
    Compute training loss.

    Args:
        est_flow: torch.Tensor, [b, n, 3]. Estimated flow.
        batch: datasets.generic.Batch. Contains ground truth flow and mask.
        feats: torch.Tensor, [b, n, c]. Features for first point cloud.
        return_fsc: boolean. Return fsc loss or not.

    Returns:
        consistent_loss: torch.Tensor, float.
        fsc_loss: torch.Tenor, float. Only returned when `return_fsc` is True.
    """
    mask = batch["mask"][..., 0]
    valid = mask > 0

    # supervised loss
    true_flow = batch["flow"]
    error = est_flow - true_flow
    error = error[valid]
    consistent_loss = torch.mean(torch.abs(error))

    # fsc loss
    fsc_loss = None
    if return_fsc:
        nb_neighbors = 32
        graph = Graph.construct_graph(batch["pc0"], nb_neighbors)
        fsc_loss = compute_fsc_loss(est_flow, feats, graph)  # [b, n]
        fsc_loss = fsc_loss[valid]
        fsc_loss = torch.mean(fsc_loss)

    if return_fsc:
        return consistent_loss, fsc_loss
    else:
        return consistent_loss


def print_rank0(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def train_ddp_worker(rank, world_size, dataset_name, nb_iter, batch_size, max_points, nb_epochs, ckpt_path, port_num, voxel_size):
    """
    Entry point of the script.

    Args:
        rank: index of the current process
        world_size: total number of the process
        dataset_name: dataset name for training
        nb_iter: number of iterations to unroll in the Sinkhorn algorithm.
        batch_size: batch size for training.
        max_points: point number used for training in each point clould.
        nb_epochs: total epoch number for training.
        ckpt_path: path for the pretrained model.
        port_num: port number for the master process in the DDP process group.
        voxel_size: Voxel size when doing voxelization.

    Returns:
        None. Model will be saved in `experiments` directory.
    """

    # ---- 1. setup process groups
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port_num
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # ---- 2. setup mp_model and devices for this process

    # Path to current file
    pathroot = os.path.dirname(__file__)

    # Path to dataset
    if dataset_name.lower() == "HPLFlowNet".lower():
        path2data = os.path.join(
            pathroot, "data", "HPLFlowNet", "FlyingThings3D_subset_processed_35m"
        )
        from datasets.flyingthings3d_hplflownet import FT3D

        lr_lambda = lambda epoch: 1.0 if epoch < 50 else 0.1
    else:
        raise ValueError("Invalid dataset name: " + dataset_name)

    # Training dataset
    ft3d_train = FT3D(root_dir=path2data, nb_points=max_points, mode="train", voxel_size=voxel_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(ft3d_train)
    trainloader = DataLoader(
        ft3d_train,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=10,
        collate_fn=Batch,
        drop_last=True,
        sampler=train_sampler,
    )

    # Model
    torch.cuda.set_device(rank)
    scene_flow = SCTN(nb_iter=nb_iter, voxel_size=voxel_size).to(rank)
    scene_flow = torch.nn.parallel.DistributedDataParallel(scene_flow, device_ids=[rank])

    # Optimizer
    optimizer = torch.optim.Adam(scene_flow.parameters(), lr=1e-3)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Log directory
    now = datetime.now().strftime("%y_%m_%d-%H_%M_%S_%f")
    now += "__Iter_" + str(nb_iter)
    now += "__Pts_" + str(max_points)
    path2log = os.path.join(pathroot, "experiments", "logs_" + dataset_name, now)

    # Load checkpoint. Suppose the checkpoint file name format is model-{epoch}.tar
    start_epoch = 0
    if ckpt_path is not None:
        try:
            start_epoch = int(os.path.basename(ckpt_path)[6:][:-4]) + 1
            print_rank0(rank, "parsed start_epoch is: {}".format(start_epoch))
        except:
            pass

        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        file = torch.load(ckpt_path, map_location=map_location)
        optimizer.load_state_dict(file['optimizer'])
        scheduler.load_state_dict(file['scheduler'])

        # add module prefix for each key
        new_state_dict = OrderedDict()
        for k, v in file['model'].items():
            name = "module." + k
            new_state_dict[name] = v
        scene_flow.load_state_dict(new_state_dict)

        print_rank0(rank, "load checkpoint from: {}".format(ckpt_path))

    # Train
    print_rank0(rank, "Training started. Logs in " + path2log)
    train_ddp(rank, world_size, scene_flow, trainloader, 10, optimizer, scheduler, path2log, nb_epochs, train_sampler, start_epoch)

    # ----- -1. cleanup process groups
    dist.destroy_process_group()


def train_ddp(rank, world_size, scene_flow, trainloader, delta, optimizer, scheduler, path2log, nb_epochs, sampler, start_epoch):
    """

    Args:
        rank: Index of the current process
        world_size: Total number of the process
        scene_flow: SCTN model
        trainloader: Data loader for training
        delta: Frequency of logs in number of iterations.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        path2log: Directory for saving log and models.
        nb_epochs: Total epoch number for training.
        sampler: Distributed sampler used in training.
        start_epoch: Epoch number when start the training process.

    Returns:
        None
    """
    print_rank0(rank, 'rank: {}/{}'.format(rank, world_size))

    # Log directory
    if rank == 0:
        if not os.path.exists(path2log):
            os.makedirs(path2log)
        writer = SummaryWriter(path2log)

    # Reload state
    total_it = 0
    epoch_start = start_epoch

    # Train
    epoch_iter_count = len(trainloader)
    for epoch in range(epoch_start, nb_epochs + epoch_start):

        # Call set_epoch to shuffle data in each epoch
        sampler.set_epoch(epoch)

        # Init.
        running_epe = 0
        running_loss = 0

        # Train for 1 epoch
        start = time.time()

        print_rank0(rank, f"Epoch: {epoch + 1}/{nb_epochs}")

        for it, batch in enumerate(trainloader):
            iter_start_time = time.time()

            # Send data to GPU
            batch = batch.to(rank, non_blocking=True)

            optimizer.zero_grad()

            # Forward
            est_flow, feats_0 = scene_flow(batch["pc0"], batch["pc1"], batch["s_C"], batch["s_F"],
                                           batch["t_C"], batch["t_F"], return_feats=True)

            if epoch < 40:
                consistent_loss = compute_loss(est_flow, batch, feats_0, return_fsc=False)
                fsc_loss = torch.zeros(1, device=consistent_loss.device)
            else:
                consistent_loss, fsc_loss = compute_loss(est_flow, batch, feats_0, return_fsc=True)

            loss = consistent_loss + 0.30 * fsc_loss

            # Backward
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            # Loss
            running_loss += loss.item()
            current_epe = compute_epe(est_flow, batch).item()
            running_epe += current_epe

            total_it += 1

            if (it + 1) % delta == 0:
                print_rank0(rank, f"epoch:{epoch + 1}, "
                                  f"iter:{it + 1}/{epoch_iter_count}, "
                                  f"{time.time() - iter_start_time: .2f} sec/iter, "
                                  f"loss:{loss.item(): .6f}, "
                                  f"c_loss:{consistent_loss.item(): .6f}, "
                                  f"f_loss:{fsc_loss.item(): .6f}, "
                                  f"epe:{current_epe: .6f}")

        # Update learning rate
        scheduler.step()

        # Save log and model
        if rank == 0:
            # Print / save logs
            writer.add_scalar("Loss/epe", running_epe / epoch_iter_count, total_it)
            writer.add_scalar("Loss/loss", running_loss / epoch_iter_count, total_it)
            print(rank, "Epoch {0:d} - It. {1:d}: loss = {2:e}".format(
                    epoch, total_it, running_loss / epoch_iter_count
            ), flush=True)
            print("{} seconds".format(time.time() - start), flush=True)

            # Save model after each epoch
            state = {
                "nb_iter": scene_flow.module.nb_iter,
                "model": scene_flow.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(state, os.path.join(path2log, "model-{}.tar".format(epoch)))

    print_rank0(rank, "Finish Training")


if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser(description="Train SCTN.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HPLFlowNet",
        help="Training dataset. Only support HPLFlowNet",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--nb_epochs", type=int, default=40, help="Number of epochs.")
    parser.add_argument(
        "--nb_points",
        type=int,
        default=8192,
        help="Maximum number of points in point cloud.",
    )
    parser.add_argument(
        "--nb_iter",
        type=int,
        default=1,
        help="Number of unrolled iterations of the Sinkhorn " + "algorithm.",
    )

    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.07,
        help="voxel size",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="check point path to load",
    )

    parser.add_argument(
        "--port_num",
        type=str,
        default='12355',
        help="localhost port number to init process group",
    )

    args = parser.parse_args()

    # Launch training using DDP (single node, multi GPU).
    n_gpus = torch.cuda.device_count()
    assert n_gpus > 0, f"Requires at least 1 GPUs to run, but got {n_gpus}."
    mp.spawn(train_ddp_worker, nprocs=n_gpus, args=(
        n_gpus,
        args.dataset,
        args.nb_iter,
        args.batch_size,
        args.nb_points,
        args.nb_epochs,
        args.ckpt_path,
        args.port_num,
        args.voxel_size
    ))
