import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_config


class VoxelLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(VoxelLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.small_addon = 1e-6

        self.config = load_config()

    def forward(self, prob_score_map, reg_map,
                pos_equal_one, neg_equal_one, targets):
        # [BS, C, W, H] -> [BS, W, H, C]
        prob_score_map = prob_score_map.permute(
            0, 2, 3, 1).contiguous()
        reg_map = reg_map.permute(
            0, 2, 3, 1).contiguous()

        width, height = reg_map.shape[1], reg_map.shape[2]

        # [BS, W, H, 14] -> [BS, W, H, 2, 7]
        reg_map = reg_map.view(
            -1, width, height, self.config['anchors_per_position'], 7)
        targets = targets.view(
            -1, width, height,  self.config['anchors_per_position'], 7)

        # Network output for positive anchor aₚₒₛ and negative anchor aₙₑ
        pos_anchor_predictions = prob_score_map[pos_equal_one == 1]
        neg_anchor_predictions = prob_score_map[neg_equal_one == 1]

        # (1 / Nₚₒₛ) ∑ BCE(pᵖᵒˢ, 1)
        bce_pos_loss = F.binary_cross_entropy(
            input=pos_anchor_predictions,
            target=torch.ones_like(pos_anchor_predictions), reduction='sum') \
            / (pos_equal_one.sum() + self.small_addon)

        # (1 / Nₙₑ) ∑ BCE(pₙₑ, 0)
        bce_neg_loss = F.binary_cross_entropy(
            input=neg_anchor_predictions,
            target=torch.zeros_like(neg_anchor_predictions), reduction='sum') \
            / (neg_equal_one.sum() + self.small_addon)

        bce_total = (self.alpha * bce_pos_loss) + (self.beta * bce_neg_loss)

        # 𝘂ᵢ and 𝘂ᵢ* are  the  regression  output and ground truth for
        # positive anchor aᵖᵒˢ
        u = reg_map[pos_equal_one == 1]
        u_star = targets[pos_equal_one == 1]

        # (1 / Nₚₒₛ) ∑ REG(𝘂ᵢ, 𝘂ᵢ*)
        reg_loss = F.smooth_l1_loss(
            input=u, target=u_star, reduction='sum') \
            / (pos_equal_one.sum() + self.small_addon)

        total_loss = bce_total + reg_loss
        return total_loss, bce_total, reg_loss
