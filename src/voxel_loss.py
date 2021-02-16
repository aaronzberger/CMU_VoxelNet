import torch
import torch.nn as nn


class VoxelLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(VoxelLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(reduction='sum')
        self.alpha = alpha
        self.beta = beta

    def forward(self, prob_score_map, reg_map,
                pos_equal_one, neg_equal_one, targets):
        # [BS, 2, X, Y] -> [BS, X, Y, 2]
        p_pos = torch.sigmoid(prob_score_map.permute(0, 2, 3, 1))

        # [BS, 14, X, Y] -> [BS, X, Y, 14]
        reg_map = reg_map.permute(0, 2, 3, 1).contiguous()

        # [BS, X, Y, 14] -> [BS, X, Y, 2, 7]
        reg_map = reg_map.view(
            reg_map.size(0), reg_map.size(1), reg_map.size(2), -1, 7)
        targets = targets.view(
            targets.size(0), targets.size(1), targets.size(2), -1, 7)

        # [BS, X, Y, 2, 7]
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(
            pos_equal_one.dim()).expand(-1, -1, -1, -1, 7)

        reg_map_pos = reg_map * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg

        # (1/Nₚₒₛ) ∑ BCE(pᵖᵒˢ, 1)
        cls_pos_loss = -pos_equal_one * torch.log(p_pos + 1e-6)
        cls_pos_loss = cls_pos_loss.sum() / (pos_equal_one.sum() + 1e-6)

        # (1/Nₙₑ) ∑ BCE(pₙₑ, 0)
        cls_neg_loss = -neg_equal_one * torch.log(1 - p_pos + 1e-6)
        cls_neg_loss = cls_neg_loss.sum() / (neg_equal_one.sum() + 1e-6)

        conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss

        # (1/Nₚₒₛ) ∑ REG(𝘂ᵢ, 𝘂ᵢ﹡)
        reg_loss = self.smoothl1loss(reg_map_pos, targets_pos)
        reg_loss = reg_loss / (pos_equal_one.sum() + 1e-6)

        total_loss = conf_loss + reg_loss
        return total_loss, conf_loss, reg_loss
