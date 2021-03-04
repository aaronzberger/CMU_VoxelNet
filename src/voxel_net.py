import torch.nn as nn
import torch
from utils import load_config, get_num_voxels

config = load_config()


class FCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCN, self).__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # View as list of point features
        num_voxels, max_pts, _ = x.shape
        x = x.view(num_voxels * max_pts, -1)

        # Each point is transformed through the fully connected network (FCN)
        # The FCN is composed of a linear layer, a batch normalization (BN)
        # layer, and a rectified linear unit (ReLU) layer
        # (2.1.1 Stacked Voxel Feature Encoding)
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        # View as voxels
        x = x.view(num_voxels, max_pts, -1)

        return x


# conv2d + bn + relu
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p,
                 activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.batch_norm = batch_norm
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=k, stride=s, padding=p
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = self.relu(x)
        return x


# conv3d + bn + relu
class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.batch_norm = batch_norm
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=k, stride=s, padding=p
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.relu(x)
        return x


# Single Voxel Feature Encoding Layer
class VFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VFE, self).__init__()
        self.in_channels = in_channels

        # The linear layer learns a matrix of size cᵢₙ×(cₒᵤₜ/2)
        # (2.1.1 Stacked Voxel Feature Encoding)
        assert out_channels % 2 == 0
        self.units = out_channels // 2
        self.fcn = FCN(self.in_channels, self.units)

    def forward(self, x, mask):
        # Each pt is transformed through the FCN into a feature space (2.1.1)
        x = self.fcn(x)

        # After obtaining point-wise feature representations, we use
        # elementwise MaxPooling across all points get the
        # locally aggregated feature (2.1.1)
        aggregated = torch.max(x, dim=1, keepdim=True)[0]

        # Augment each point with the aggregated feature to form the
        # point-wise concatenated feature (2.1.1)
        repeated = aggregated.expand(-1, config['max_pts_per_voxel'], -1)
        concat = torch.cat([x, repeated], dim=2)

        mask = mask.expand(-1, -1, self.units * 2)

        # mask encodes where the points actually exist, so multiply by mask
        # to filter out where we concatenated to non-existent points above
        concat = concat * mask.float()

        return concat


# Stacked Voxel Feature Encoding
class SVFE(nn.Module):
    def __init__(self):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(7, 32)
        self.vfe_2 = VFE(32, 128)
        self.fcn = FCN(128, 128)

    def forward(self, x):
        # mask is a (X, 35, 1) and represents for each voxel,
        # for each possible point, whether a point exists there
        # torch.max returns (values, indices), so use the first
        mask = torch.ne(torch.max(x, dim=2, keepdim=True)[0], 0)

        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)

        # The voxel-wise feature is obtained by transforming the output of
        # VFE-n through an FCN and applying element-wise Maxpool (2.1.1)
        x = self.fcn(x)
        voxelwise_features = torch.max(x, dim=1)[0]

        return voxelwise_features


# Convolutional Middle Layer
class CML(nn.Module):
    def __init__(self):
        super(CML, self).__init__()
        self.conv3d_1 = Conv3d(128, 64, 3, s=(2, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(0, 1, 1))
        self.conv3d_3 = Conv3d(64, 64, 3, s=(2, 1, 1), p=(1, 1, 1))

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x


# Region Proposal Network
class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.block_1 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_1 += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_2 += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv2d(128, 256, 3, 2, 1)]
        self.block_3 += [nn.Conv2d(256, 256, 3, 1, 1) for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 4, 0),
            nn.BatchNorm2d(256), nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 2, 2, 0),
            nn.BatchNorm2d(256), nn.ReLU()
        )

        # Paper specifies padding as 0, but that gives dimension mismatch error
        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256), nn.ReLU()
        )

        self.prob_map = Conv2d(768, config["anchors_per_position"],
                               1, 1, 0, activation=False, batch_norm=False)

        self.reg_map = Conv2d(768, 7 * config["anchors_per_position"],
                              1, 1, 0, activation=False, batch_norm=False)

    def forward(self, x):
        x = self.block_1(x)
        block_1_feature_map = x
        x = self.block_2(x)
        block_2_feature_map = x
        x = self.block_3(x)

        # Deconv output of each block to concat into high-res feature map
        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(block_2_feature_map)
        x_2 = self.deconv_3(block_1_feature_map)

        x = torch.cat((x_0, x_1, x_2), dim=1)

        # Turn high-res feature map into probability score map and reg map
        return self.prob_map(x), self.reg_map(x)


class VoxelNet(nn.Module):
    def __init__(self, device):
        super(VoxelNet, self).__init__()
        self.svfe = SVFE().to(device)
        self.cml = CML().to(device)
        self.rpn = RPN().to(device)
        self.device = device
        self.voxel_W, self.voxel_H, self.voxel_D = get_num_voxels()

        def init_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight.data)

        self.apply(init_weights)

    def voxel_indexing(self, sparse_features, coords):
        '''
        More than 90% of voxels are empty, so in the Voxel Feature Encoding
        Layers, we use sparse tensor format. Now, we have to convert back
        for the CML and RPN.

        See 2.1.1: Sparse Tensor Representation

        Parameters:
            sparse_features (arr): output from the SVFE
            coords (arr): the coordinates of the non-empty voxels so we
                can put the sparse features into the correct places in the
                new dense tensor
        '''
        dim = sparse_features.shape[-1]

        # # This uses the PyTorch Sparse library
        # sparse = torch.sparse.FloatTensor(
        #     coords.t(),
        #     sparse_features,
        #     torch.Size([config["batch_size"], self.voxel_D,
        #                 self.voxel_H, sparse_features, dim]),
        # )

        # dense = sparse.to_dense()

        # return dense

        coords = coords.type(torch.LongTensor)

        dense_feature = torch.zeros(
            dim, config["batch_size"], self.voxel_D, self.voxel_H, self.voxel_W
        ).to(self.device)

        dense_feature[
            :, coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        ] = sparse_features.transpose(0, 1)

        return dense_feature.transpose(0, 1)

    def forward(self, voxel_features, voxel_coords):
        """
        Parameters:
            voxel_features (arr): (BS, N, X, 7),
                where N = number of non-empty voxels,
                X = max points per voxel (See T in 2.1.1), and
                7 encodes [x,y,z,r,Δx,Δy,Δz]
            voxel_coords (arr): (BS, N, 4),
                where N = number of non-empty voxels and
                4 encodes [Batch ID, X voxel, Y voxel, Z voxel]
        """
        # Stacked Voxel Feature Encoding Layers
        vfe_output = self.svfe(voxel_features)
        indexed = self.voxel_indexing(vfe_output, voxel_coords)

        # Convolutional Middle Layer
        cml_output = self.cml(indexed)

        # Region Proposal Network
        prob_score_map, reg_map = self.rpn(
            cml_output.view(
                config['batch_size'], -1, self.voxel_H, self.voxel_W))

        return torch.sigmoid(prob_score_map), reg_map
