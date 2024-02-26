import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

import utils.general as utils


class VolSDFLoss(nn.Module):
    def __init__(self, rgb_loss, eikonal_weight, **kargs):
        super().__init__()

        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')
        self.eikonal_weight = eikonal_weight
        self.fg_mask_loss_weight = kargs.get('fg_mask_loss_weight', 0)
        self.loss_rgb_weight = kargs.get('loss_rgb_weight', 1.0)



    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def forward(self, model_outputs, ground_truth, **kargs):
        rgb_gt = ground_truth['rgb'].cuda()
        eps = 1e-5
        output = {"loss":None}
        if 'rgb_values' in model_outputs and model_outputs['rgb_values'] is not None:
            rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt.reshape([-1, 3]))
            output['rgb_loss'] = rgb_loss
        else:
            rgb_loss = torch.tensor(0.0).cuda().float()

        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
            output['eikonal_loss'] = eikonal_loss
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        if 'weights' in model_outputs and model_outputs['weights'] is not None:
            fg_mask = 1 - ground_truth['bg_mask'].squeeze().to(device=rgb_gt.device,dtype=torch.float32)
            weights = model_outputs['weights'].sum(dim=-1)
            fg_mask_loss = F.binary_cross_entropy(weights.clip(1e-3, 1.0 - 1e-3), fg_mask)
            output['fg_mask_loss'] = fg_mask_loss
        else:
            fg_mask_loss = torch.tensor(0.0).cuda().float()

        loss = self.eikonal_weight * eikonal_loss + \
               self.fg_mask_loss_weight * fg_mask_loss + \
               self.loss_rgb_weight * rgb_loss

        output['loss'] = loss
        return output
