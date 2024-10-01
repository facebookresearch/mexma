# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from DinoV2 implementation at:
# https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/loss/koleo_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)
        return I

    def forward(self, cls_embeddings, eps=1e-8):
        """
        Args:
            cls_embeddings (B, E): cls embeddings from the encoder
        """
        with torch.cuda.amp.autocast(enabled=False):
            cls_embeddings = F.normalize(cls_embeddings, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(cls_embeddings)
            distances = self.pdist(cls_embeddings, cls_embeddings[I])
            loss = -torch.log(distances + eps).mean()
        return loss
