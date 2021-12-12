import torch

from .transformer import TransformerBlock
from .ot import sinkhorn
from .graph import Graph
from .gconv import SetConv
from .minkowski import MinkowskiFeatureExtractor
import MinkowskiEngine as ME


class SCTN(torch.nn.Module):
    def __init__(self, nb_iter, voxel_size=0.05):
        """
        Construct a model that, once trained, estimate the scene flow between
        two point clouds.

        Args:
            nb_iter: int. Number of iterations to unroll in the Sinkhorn algorithm.
            voxel_size: float. Voxel size when do voxelization.
        """
        super(SCTN, self).__init__()

        # Output feature channels for each point.
        n = 32

        # OT parameters. Number of unrolled iterations in the Sinkhorn algorithm
        self.nb_iter = nb_iter

        # Mass regularisation
        self.gamma = torch.nn.Parameter(torch.zeros(1))

        # Entropic regularisation
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))

        self.minkowski_extractor = MinkowskiFeatureExtractor(voxel_size=voxel_size)

        self.pct = TransformerBlock(64)

        # Refinement
        self.ref_conv1 = SetConv(3, n)
        self.ref_conv2 = SetConv(n, 2 * n)
        self.ref_conv3 = SetConv(2 * n, 4 * n)
        self.fc = torch.nn.Linear(4 * n, 3)

    def refine(self, flow, graph):
        """
        Refine the input flow thanks to a residual network.

        Args:
            flow: torch.Tensor, [B, N, 3]. Input flow to refine.
            graph: models.Graph. Graph build on the point cloud on which the flow is defined.

        Returns:
            refined_flow: [B, N, 3]
        """

        x = self.ref_conv1(flow, graph)
        x = self.ref_conv2(x, graph)
        x = self.ref_conv3(x, graph)
        x = self.fc(x)

        return flow + x

    def forward(self, pc0, pc1, s_C, s_F, t_C, t_F, return_feats=False):
        """
        Estimate scene flow between two input point clouds.

        Args:
            pc0: [b, n, c]. Original point coordinates for pc0.
            pc1: [b, n, c]. Original point coordinates for pc1.
            s_C: All voxel's coordinates for pc0.
            s_F: All voxel's initial feature for pc0
            t_C: All voxel's coordinates for pc1.
            t_F: All voxel's initial feature for pc1.
            return_feats: Whether return features for each point in pc0 or not.

        Returns:
            refined_flow: [B, N, 3]. Final estimated flow for each point in pc0.
            fused_feats_0: [B, N, C]. Features for each point in pc0. Only returned when `return_feats` is True.
        """

        # Computing features by using sparse convolution.
        sinput_0 = ME.SparseTensor(features=s_F, coordinates=s_C)
        sinput_1 = ME.SparseTensor(features=t_F, coordinates=t_C)
        upsample_n = 32
        feats_0, feats_1 = self.minkowski_extractor(sinput_0, sinput_1, pc0, pc1, upsample_n)

        # Refine features by using transformer.
        sa_feats_0 = self.pct(pc0, feats_0)
        sa_feats_1 = self.pct(pc1, feats_1)

        # Feature fusion
        fused_feats_0 = feats_0 + sa_feats_0
        fused_feats_1 = feats_1 + sa_feats_1

        # Estimate coarse flow with OT
        transport = sinkhorn(
            fused_feats_0,
            fused_feats_1,
            pc0,
            pc1,
            epsilon=torch.exp(self.epsilon) + 0.025,
            gamma=torch.exp(self.gamma),
            max_iter=self.nb_iter,
        )
        row_sum = transport.sum(-1, keepdim=True)
        ot_flow = (transport @ pc1) / (row_sum + 1e-8) - pc0

        # Flow refinement
        graph0 = Graph.construct_graph(pc0, 32)
        refined_flow = self.refine(ot_flow, graph0)

        if return_feats:
            return refined_flow, fused_feats_0
        else:
            return refined_flow
