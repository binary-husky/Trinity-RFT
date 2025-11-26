"""PPO policy loss function.

Modified from https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
"""

from typing import Dict, Optional, Tuple

import torch
import numpy as np

from trinity.algorithm.policy_loss_fn.policy_loss_fn import POLICY_LOSS_FN, PolicyLossFn
from trinity.algorithm.utils import masked_loss, masked_mean


@POLICY_LOSS_FN.register_module("ppo")
class PPOPolicyLossFn(PolicyLossFn):
    def __init__(
        self,
        backend: str = "verl",
        clip_range: Optional[float] = None,
        clip_range_low: Optional[float] = None,
        clip_range_high: Optional[float] = None,
        loss_agg_mode: Optional[str] = "token-mean",
    ) -> None:
        super().__init__(backend=backend)
        if clip_range_low is None:
            self.clip_range_low = clip_range
        else:
            self.clip_range_low = clip_range_low
        if clip_range_high is None:
            self.clip_range_high = clip_range
        else:
            self.clip_range_high = clip_range_high
        assert self.clip_range_low is not None, "clip_range_low must be specified."
        assert self.clip_range_high is not None, "clip_range_high must be specified."
        self.loss_agg_mode = loss_agg_mode

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:

        E = logprob - old_logprob
        E_clip = torch.zeros_like(E)
        E_clip = torch.where(advantages > 0, torch.clamp(E, max=np.log(1.0+self.clip_range_high)), E_clip)
        E_clip = torch.where(advantages < 0, torch.clamp(E, min=np.log(1.0-self.clip_range_low), max=np.log(5) ), E_clip)
        ratio = torch.exp(E_clip)
        print('using dual clip ppo loss')
        pg_losses_comb = -advantages * ratio
        pg_loss = masked_loss(
            pg_losses_comb, action_mask, loss_agg_mode=self.loss_agg_mode
        )

        # negative_approx_kl = logprob - old_logprob
        # ratio = torch.exp(negative_approx_kl)
        ppo_kl = masked_mean(-E, action_mask)

        # pg_losses = -advantages * ratio
        # pg_losses2 = -advantages * torch.clamp(
        #     ratio, 1.0 - self.clip_range_low, 1.0 + self.clip_range_high  # type: ignore
        # )

        # pg_loss = masked_loss(
        #     torch.max(pg_losses, pg_losses2), action_mask, loss_agg_mode=self.loss_agg_mode
        # )
        pg_clipfrac = masked_mean((E_clip == E).float(), action_mask)
        metrics = {
            "pg_clipfrac": pg_clipfrac.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "pg_loss": pg_loss.detach().item(),
        }
        return pg_loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "clip_range": 0.2,
            "loss_agg_mode": "token-mean",
        }
