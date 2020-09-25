import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli


class DropBlock(nn.Module):
    def __init__(self, drop_prob, block_size, warmup_steps=2500):
        super().__init__()
        self.block_size = block_size
        self.current_step = 0
        self.drop_values = np.linspace(
            start=1e-6, stop=drop_prob, num=int(warmup_steps)
        )
        self.drop_prob = self.drop_values[0]

    def step(self):
        if self.current_step < len(self.drop_values):
            self.drop_prob = self.drop_values[self.current_step]
            self.current_step += 1

    def forward(self, x):
        if self.training:
            batch_size, channels, height, width = x.size()

            gamma = self._compute_gamma(x)
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (
                    batch_size,
                    channels,
                    height - (self.block_size - 1),
                    width - (self.block_size - 1),
                )
            ).to(x.device)

            block_mask = self._compute_block_mask(mask)
            scale = torch.numel(block_mask) / block_mask.sum()

            return x * block_mask * scale
        else:
            return x

    def _compute_gamma(self, x):
        feat_size = x.size(2)
        gamma = (
            self.drop_prob
            / self.block_size ** 2
            * feat_size ** 2
            / (feat_size - self.block_size + 1) ** 2
        )
        return gamma

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.size()
        non_zero_idxs = torch.nonzero(mask, as_tuple=False)
        nr_blocks = non_zero_idxs.size(0)

        offsets = (
            torch.stack(
                [
                    torch.arange(self.block_size)
                    .view(-1, 1)
                    .expand(self.block_size, self.block_size)
                    .reshape(-1),
                    torch.arange(self.block_size).repeat(self.block_size),
                ]
            )
            .t()
            .to(mask.device)
        )
        offsets = torch.cat(
            (
                torch.zeros(self.block_size ** 2, 2, device=mask.device).long(),
                offsets.long(),
            ),
            1,
        )

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(
                mask, (left_padding, right_padding, left_padding, right_padding)
            )
            padded_mask[
                block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]
            ] = 1.0
        else:
            padded_mask = F.pad(
                mask, (left_padding, right_padding, left_padding, right_padding)
            )

        block_mask = 1 - padded_mask
        return block_mask
