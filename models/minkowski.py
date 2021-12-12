import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF

_EPS = 1e-6


class ResBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = 'BN'

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 D=3):
        super(ResBlockBase, self).__init__()

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dimension=D)

        self.norm1 = get_norm_layer(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)

        self.conv2 = ME.MinkowskiConvolution(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            dimension=D)

        self.norm2 = get_norm_layer(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = MEF.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = MEF.relu(out)

        return out


class ResBlockBN(ResBlockBase):
    NORM_TYPE = 'BN'


class ResBlockIN(ResBlockBase):
    NORM_TYPE = 'IN'


def get_res_block(norm_type,
                  inplanes,
                  planes,
                  stride=1,
                  dilation=1,
                  downsample=None,
                  bn_momentum=0.1,
                  D=3):
    if norm_type == 'BN':
        return ResBlockBN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)

    elif norm_type == 'IN':
        return ResBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)

    else:
        raise ValueError(f'Type {norm_type}, not defined')


def get_norm_layer(norm_type, num_feats, bn_momentum=0.05, D=-1):
    if norm_type == 'BN':
        return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)

    elif norm_type == 'IN':
        return ME.MinkowskiInstanceNorm(num_feats)

    else:
        raise ValueError(f'Type {norm_type}, not defined')


class SparseEnoder(ME.MinkowskiNetwork):
    CHANNELS = [None, 64, 64, 128, 128]

    def __init__(self,
                 in_channels=3,
                 out_channels=128,
                 bn_momentum=0.1,
                 conv1_kernel_size=9,
                 norm_type='IN',
                 D=3):
        ME.MinkowskiNetwork.__init__(self, D)

        NORM_TYPE = norm_type
        BLOCK_NORM_TYPE = norm_type
        CHANNELS = self.CHANNELS

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[1],
            kernel_size=conv1_kernel_size,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm1 = get_norm_layer(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.block1 = get_res_block(
            BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)

        self.norm2 = get_norm_layer(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2 = get_res_block(
            BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3 = get_norm_layer(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3 = get_res_block(
            BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv4 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[3],
            out_channels=CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm4 = get_norm_layer(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.block4 = get_res_block(
            BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

    def forward(self, x, tgt_feature=False):
        skip_features = []
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out = self.block1(out_s1)

        skip_features.append(out_s1)

        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out = self.block2(out_s2)

        skip_features.append(out_s2)

        out_s4 = self.conv3(out)
        out_s4 = self.norm3(out_s4)
        out = self.block3(out_s4)

        skip_features.append(out_s4)

        out_s8 = self.conv4(out)
        out_s8 = self.norm4(out_s8)
        out = self.block4(out_s8)

        return out, skip_features


class SparseDecoder(ME.MinkowskiNetwork):
    TR_CHANNELS = [None, 64, 128, 128, 128]
    CHANNELS = [None, 64, 64, 128, 128]

    def __init__(self,
                 out_channels=128,
                 bn_momentum=0.1,
                 norm_type='IN',
                 D=3):
        ME.MinkowskiNetwork.__init__(self, D)

        NORM_TYPE = norm_type
        BLOCK_NORM_TYPE = norm_type
        TR_CHANNELS = self.TR_CHANNELS
        CHANNELS = self.CHANNELS

        self.conv4_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[4],
            out_channels=TR_CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)

        self.norm4_tr = get_norm_layer(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.block4_tr = get_res_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.conv3_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[3] + TR_CHANNELS[4],
            out_channels=TR_CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3_tr = get_norm_layer(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3_tr = get_res_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv2_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[2] + TR_CHANNELS[3],
            out_channels=TR_CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2_tr = get_norm_layer(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2_tr = get_res_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv1_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[1] + TR_CHANNELS[2],
            out_channels=TR_CHANNELS[1],
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)

        self.final = ME.MinkowskiConvolution(
            in_channels=TR_CHANNELS[1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=D)

    def forward(self, x, skip_features):
        out = self.conv4_tr(x)
        out = self.norm4_tr(out)

        out_s4_tr = self.block4_tr(out)

        out = ME.cat(out_s4_tr, skip_features[-1])

        out = self.conv3_tr(out)
        out = self.norm3_tr(out)
        out_s2_tr = self.block3_tr(out)

        out = ME.cat(out_s2_tr, skip_features[-2])

        out = self.conv2_tr(out)
        out = self.norm2_tr(out)
        out_s1_tr = self.block2_tr(out)

        out = ME.cat(out_s1_tr, skip_features[-3])

        out = self.conv1_tr(out)
        out = MEF.relu(out)
        out = self.final(out)

        return out


class MinkowskiFeatureExtractor(nn.Module):
    def __init__(self, voxel_size, normalize_feature=True):
        super(MinkowskiFeatureExtractor, self).__init__()

        self.voxel_size = voxel_size
        self.normalize_feature = normalize_feature

        self.input_feature_dim = 3

        # Initialize the backbone network
        self.encoder = SparseEnoder(in_channels=self.input_feature_dim,
                                    conv1_kernel_size=7,
                                    norm_type="IN")

        self.decoder = SparseDecoder(out_channels=64,
                                     norm_type="IN")

    def forward(self, st_1, st_2, xyz_1, xyz_2, k_values):
        """
        Computing feature for each point in pc1 and pc2 using minkowski engine based network.

        Args:
            st_1: sparse tensor for pc1
            st_2: sparse tensor for pc2
            xyz_1: original coordinates for pc1, [b, n, 3]
            xyz_2: original coordinates for pc2, [b, n, 3]
            k_values: neighbor count used when doing upsample.

        Returns:
            pc1_feature: Features for each original point. [b, n, c]
            pc2_feature: Features for each original point. [b, n, c]

        """

        # Run both point clouds through the backbone network
        enc_feat_1, skip_features_1 = self.encoder(st_1)
        enc_feat_2, skip_features_2 = self.encoder(st_2)

        dec_feat_1 = self.decoder(enc_feat_1, skip_features_1)
        dec_feat_2 = self.decoder(enc_feat_2, skip_features_2)

        # check normalization !!!!!
        if self.normalize_feature:
            dec_feat_1 = ME.SparseTensor(
                        dec_feat_1.F / torch.norm(dec_feat_1.F, p=2, dim=1, keepdim=True),
                        coordinate_map_key=dec_feat_1.coordinate_map_key,
                        coordinate_manager=dec_feat_1.coordinate_manager)

            dec_feat_2 = ME.SparseTensor(
                        dec_feat_2.F / torch.norm(dec_feat_2.F, p=2, dim=1, keepdim=True),
                        coordinate_map_key=dec_feat_2.coordinate_map_key,
                        coordinate_manager=dec_feat_2.coordinate_manager)

        # Upsample
        up_f_1 = self.upsample(xyz_1, dec_feat_1, k_value=k_values)
        up_f_2 = self.upsample(xyz_2, dec_feat_2, k_value=k_values)

        return torch.stack(up_f_1, dim=0), torch.stack(up_f_2, dim=0)

    def upsample(self, xyz, sparse_tensor, k_value=3):
        dense_flow = []
        b, n, _ = xyz.shape
        for b_idx in range(b):
            sparse_xyz = sparse_tensor.coordinates_at(b_idx).cuda() * self.voxel_size
            sparse_feature = sparse_tensor.features_at(b_idx)

            sqr_dist = self.pairwise_distance(xyz[b_idx], sparse_xyz, normalized=False).squeeze(0)
            sqr_dist, group_idx = torch.topk(sqr_dist, k_value, dim=-1, largest=False, sorted=False)

            dist = torch.sqrt(sqr_dist)
            norm = torch.sum(1 / (dist + 1e-7), dim=1, keepdim=True)
            weight = ((1 / (dist + 1e-7)) / norm).unsqueeze(-1)

            sparse_flow = sparse_feature[group_idx.reshape(-1), :].reshape(n, k_value, -1)

            dense_flow.append(torch.sum(weight * sparse_flow, dim=1))

        return dense_flow

    def pairwise_distance(self, src, dst, normalized=True):
        """Calculates squared Euclidean distance between each two points.
        Args:
            src (torch tensor): source data, [b, n, c]
            dst (torch tensor): target data, [b, m, c]
            normalized (bool): distance computation can be more efficient
        Returns:
            dist (torch tensor): per-point square distance, [b, n, m]
        """

        if len(src.shape) == 2:
            src = src.unsqueeze(0)
            dst = dst.unsqueeze(0)

        B, N, _ = src.shape
        _, M, _ = dst.shape

        # Minus such that smaller value still means closer
        dist = -torch.matmul(src, dst.permute(0, 2, 1))

        # If inputs are normalized just add 1 otherwise compute the norms
        if not normalized:
            dist *= 2
            dist += torch.sum(src ** 2, dim=-1)[:, :, None]
            dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

        else:
            dist += 1.0

        # Distances can get negative due to numerical precision
        dist = torch.clamp(dist, min=0.0, max=None)

        return dist
