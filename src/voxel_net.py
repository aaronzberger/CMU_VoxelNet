import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import load_config, get_num_voxels

config = load_config()


# conv2d + bn + relu
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 k, s, p, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.batch_norm = batch_norm
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.activation:
            x = F.relu(x, inplace=True)
        return x


# conv3d + bn + relu
class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.batch_norm = batch_norm
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return F.relu(x, inplace=True)


# Fully Connected Network
class FCN(nn.Module):

    def __init__(self, cin, cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        # KK is the stacked k across batch
        kk, t, _ = x.shape
        x = self.linear(x.view(kk * t, -1))
        x = self.bn(x)
        x = F.relu(x)
        return x.view(kk, t, -1)


# Single Voxel Feature Encoding Layer
class VFE(nn.Module):

    def __init__(self, cin, cout):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin, self.units)

    def forward(self, x, mask):
        # point-wise feauture
        pwf = self.fcn(x)
        # locally aggregated feature
        laf = torch.max(pwf, 1)[0].unsqueeze(1).repeat(
            1, config['max_pts_per_voxel'], 1)
        # point-wise concat feature
        pwcf = torch.cat((pwf, laf), dim=2)
        # apply mask
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        pwcf = pwcf * mask.float()

        return pwcf


# Stacked Voxel Feature Encoding
class SVFE(nn.Module):
    def __init__(self):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(7, 32)
        self.vfe_2 = VFE(32, 128)
        self.fcn = FCN(128, 128)

    def forward(self, x):
        mask = torch.ne(torch.max(x, 2)[0], 0)  # ne = not equal
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        # element-wise max pooling
        x = torch.max(x, 1)[0]
        return x


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
            nn.ConvTranspose2d(256, 256, 4, 4, 0), nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 2, 2, 0), nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 1, 1, 0), nn.BatchNorm2d(256))

        self.prob_map = Conv2d(768, config['anchors_per_position'],
                               1, 1, 0, activation=False, batch_norm=False)
        self.reg_map = Conv2d(768, 7 * config['anchors_per_position'],
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

        x = torch.cat((x_0, x_1, x_2), 1)

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
        dim = sparse_features.shape[-1]

        coords = coords.type(torch.LongTensor)

        dense_feature = torch.zeros(dim, config['batch_size'], self.voxel_D,
                                    self.voxel_H, self.voxel_W).to(self.device)

        dense_feature[:, coords[:, 0], coords[:, 1],
                      coords[:, 2], coords[:, 3]] = \
            sparse_features.transpose(0, 1)

        return dense_feature.transpose(0, 1)

    def forward(self, voxel_features, voxel_coords):
        '''
            Parameters:
                voxel_features (arr): (N, X, 7),
                    where N = number of non-empty voxels,
                    X = max points per voxel (See T in 2.1.1), and
                    7 encodes [x,y,z,r,Δx,Δy,Δz]
                voxel_coords (arr): (N, 4),
                    where N = number of non-empty voxels and
                    4 encodes [Number in Batch, X voxel, Y voxel, Z voxel]
        '''
        # Stacked Voxel Feature Encoding Layers
        vfe_output = self.svfe(voxel_features)
        indexed = self.voxel_indexing(vfe_output, voxel_coords)

        # Convolutional Middle Layer
        cml_output = self.cml(indexed)

        # Region Proposal Network
        # merge the depth and feature dim into one,
        # output probability score map and regression map
        prob_score_map, reg_map = self.rpn(
            cml_output.view(
                config['batch_size'], -1, self.voxel_H, self.voxel_W))
        return prob_score_map, reg_map
