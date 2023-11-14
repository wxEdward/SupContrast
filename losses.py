"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


import diffdist.functional as distops
import torch
import torch.distributed as dist


def pairwise_similarity(outputs, temperature=0.5, multi_gpu=False, adv_type='None'):
    '''
        Compute pairwise similarity and return the matrix
        input: aggregated outputs & temperature for scaling
        return: pairwise cosine similarity

    '''
    if multi_gpu and adv_type == 'None':

        B = int(outputs.shape[0] / 2)

        outputs_1 = outputs[0:B]
        outputs_2 = outputs[B:]

        gather_t_1 = [torch.empty_like(outputs_1) for i in range(dist.get_world_size())]
        gather_t_1 = distops.all_gather(gather_t_1, outputs_1)

        gather_t_2 = [torch.empty_like(outputs_2) for i in range(dist.get_world_size())]
        gather_t_2 = distops.all_gather(gather_t_2, outputs_2)

        outputs_1 = torch.cat(gather_t_1)
        outputs_2 = torch.cat(gather_t_2)
        outputs = torch.cat((outputs_1, outputs_2))
    elif multi_gpu and 'Rep' in adv_type:
        if adv_type == 'Rep':
            N = 3
        B = int(outputs.shape[0] / N)

        outputs_1 = outputs[0:B]
        outputs_2 = outputs[B:2 * B]
        outputs_3 = outputs[2 * B:3 * B]

        gather_t_1 = [torch.empty_like(outputs_1) for i in range(dist.get_world_size())]
        gather_t_1 = distops.all_gather(gather_t_1, outputs_1)

        gather_t_2 = [torch.empty_like(outputs_2) for i in range(dist.get_world_size())]
        gather_t_2 = distops.all_gather(gather_t_2, outputs_2)

        gather_t_3 = [torch.empty_like(outputs_3) for i in range(dist.get_world_size())]
        gather_t_3 = distops.all_gather(gather_t_3, outputs_3)

        outputs_1 = torch.cat(gather_t_1)
        outputs_2 = torch.cat(gather_t_2)
        outputs_3 = torch.cat(gather_t_3)

        if N == 3:
            outputs = torch.cat((outputs_1, outputs_2, outputs_3))

    B = outputs.shape[0]
    outputs_norm = outputs / (outputs.norm(dim=1).view(B, 1) + 1e-8)
    similarity_matrix = (1. / temperature) * torch.mm(outputs_norm, outputs_norm.transpose(0, 1).detach())

    return similarity_matrix, outputs


def NT_xent(similarity_matrix, adv_type):
    """
        Compute NT_xent loss
        input: pairwise-similarity matrix
        return: NT xent loss
    """

    N2 = len(similarity_matrix)
    if adv_type == 'None':
        N = int(len(similarity_matrix) / 2)
    elif adv_type == 'Rep':
        N = int(len(similarity_matrix) / 3)

    # Removing diagonal #
    similarity_matrix_exp = torch.exp(similarity_matrix)
    similarity_matrix_exp = similarity_matrix_exp * (1 - torch.eye(N2, N2)).cuda()

    NT_xent_loss = - torch.log(
        similarity_matrix_exp / (torch.sum(similarity_matrix_exp, dim=1).view(N2, 1) + 1e-8) + 1e-8)

    if adv_type == 'None':
        NT_xent_loss_total = (1. / float(N2)) * torch.sum(
            torch.diag(NT_xent_loss[0:N, N:]) + torch.diag(NT_xent_loss[N:, 0:N]))
    elif adv_type == 'Rep':
        NT_xent_loss_total = (1. / float(N2)) * torch.sum(
            torch.diag(NT_xent_loss[0:N, N:2 * N]) + torch.diag(NT_xent_loss[N:2 * N, 0:N])
            + torch.diag(NT_xent_loss[0:N, 2 * N:]) + torch.diag(NT_xent_loss[2 * N:, 0:N])
            + torch.diag(NT_xent_loss[N:2 * N, 2 * N:]) + torch.diag(NT_xent_loss[2 * N:, N:2 * N]))
    return NT_xent_loss_total


if __name__ =='__main__':
    a = torch.tensor([[0.1,0.2], [0.2, 0.3]])
    print(pairwise_similarity(a))