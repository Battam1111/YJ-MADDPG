"""Rule-based deterministic baseline for the HGAM environment.

Policy
------
* **MUAV (drone)**: each step pick the in-range PoI with the maximum
  ``data_remaining / distance`` score and move in its direction. If no
  PoI is within laser range, hold position (zero action).
* **CUAV (charger)**: move toward the MUAV with the lowest remaining
  electricity. Action vector is L2-normalised to unit length.

The controller has no learnable parameters, no replay buffer, and no
``update()`` step. It is a strict drop-in for
:class:`hgam.algorithms.maddpg.MADDPGController` from the perspective
of :class:`hgam.training.runner.PybulletRunner` — all the same methods
exist with the same signatures and return shapes.

Greedy serves as the deterministic-policy baseline (paper §6.2) and is
not subject to the 5-seed averaging in E0 (any seed produces the same
action sequence given the same env state).  We still run all 5 seeds
to capture stochasticity introduced by the env's random initial
positions and PoI layouts.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pybullet as p
import torch
from torch import nn

from hgam.env.utils import caculate_2D_distance


class _NullReplayMemory:
    """No-op replay buffer used by the Greedy controller.

    Mirrors the surface of ``hgam.replay.memory.ReplayMemory`` that
    :class:`hgam.training.runner.PybulletRunner` touches in its
    training loop: ``push`` is called every step, so it has to exist
    and accept any arguments.  Sampling methods are never called for
    Greedy because :meth:`GreedyController.update` is a no-op.
    """

    is_prioritized = False

    def push(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __len__(self) -> int:
        return 0


class _DummyActor(nn.Module):
    """Placeholder so ``runner.evaluate()``'s ``actor[i].eval()`` calls work.

    The runner's eval mode flips its actor modules into ``eval`` state
    before running deterministic rollouts. We don't have anything to
    flip, but we do need the ``.eval()`` method to exist and return
    something (it returns ``self`` for an ``nn.Module``).
    """

    def forward(self, x):  # pragma: no cover - never invoked
        return torch.zeros(2, device=x.device if hasattr(x, "device") else "cpu")


class _AgentGroupStub:
    """Mimic the attributes of :class:`hgam.nn.nets.MADDPGAgent` that
    runner.py reads (``actor``, ``critic``, ``target_actor``,
    ``target_critic``) so the same code path works for Greedy.
    """

    def __init__(self, n_agents: int, device: torch.device) -> None:
        self.actor = nn.ModuleList([_DummyActor().to(device) for _ in range(n_agents)])
        critic = _DummyActor().to(device)
        self.critic = critic
        self.target_actor = self.actor
        self.target_critic = critic

    def parameters(self):  # pragma: no cover - for compat with optimizer probes
        return self.actor.parameters()


class GreedyController:
    """Deterministic rule-based controller — see module docstring."""

    def __init__(
        self,
        *,
        env: Any,
        num_UAVAgents: int,
        num_chargerAgents: int,
        device: torch.device,
        dim_action: int = 2,
    ) -> None:
        self.env = env
        self.num_UAVAgents = int(num_UAVAgents)
        self.num_chargerAgents = int(num_chargerAgents)
        self.dim_UAVactions = int(dim_action)
        self.dim_UAVobs = 0  # unused but kept for runner compat
        self.device = device

        self.memory = _NullReplayMemory()
        self.UAVAgent = _AgentGroupStub(self.num_UAVAgents, device)
        self.chargerAgent = _AgentGroupStub(self.num_chargerAgents, device)

    # ---- training-loop interface ----------------------------------------

    def act(
        self,
        state_batch,
        adj,
        episode_num: int,
        episode_before_train: int,
        if_noise: bool,
    ) -> torch.Tensor:
        """Compute one step's deterministic actions.

        Arguments other than ``state_batch`` are accepted but ignored:
        Greedy has no exploration noise and no warm-up phase.
        """
        n_total = self.num_UAVAgents + self.num_chargerAgents
        actions = torch.zeros(n_total, self.dim_UAVactions, device=self.device)

        sp_data = self.env.scene.signalPointId2data
        sp_positions = {
            sp: p.getBasePositionAndOrientation(sp)[0] for sp in sp_data
        }

        # MUAV policy: greedy toward the highest-data nearby PoI.
        for i in range(self.num_UAVAgents):
            drone = self.env.robot[i]
            drone_pos = p.getBasePositionAndOrientation(drone.robot)[0]
            best_score = -1.0
            best_target = None
            for sp, data in sp_data.items():
                if data <= 0:
                    continue
                dist = caculate_2D_distance(drone_pos, sp_positions[sp])
                if dist > getattr(drone, "LASER_LENGTH", 16.0):
                    continue
                score = data / max(dist, 0.01)
                if score > best_score:
                    best_score = score
                    best_target = sp_positions[sp]
            if best_target is not None:
                direction = np.array(best_target[:2]) - np.array(drone_pos[:2])
                norm = float(np.linalg.norm(direction))
                if norm > 1e-8:
                    direction = direction / norm
                actions[i] = torch.from_numpy(direction.astype(np.float32)).to(self.device)

        # CUAV policy: greedy toward the MUAV with lowest remaining battery.
        for j in range(self.num_chargerAgents):
            charger = self.env.charger[j]
            charger_pos = p.getBasePositionAndOrientation(charger.robot)[0]
            lowest = float("inf")
            target_pos = None
            for drone in self.env.robot:
                if drone.electricity < lowest:
                    lowest = float(drone.electricity)
                    target_pos = p.getBasePositionAndOrientation(drone.robot)[0]
            if target_pos is not None:
                direction = np.array(target_pos[:2]) - np.array(charger_pos[:2])
                norm = float(np.linalg.norm(direction))
                if norm > 1e-8:
                    direction = direction / norm
                actions[self.num_UAVAgents + j] = torch.from_numpy(
                    direction.astype(np.float32)
                ).to(self.device)

        return actions

    def update(self, i_step: int, param_dict: dict):
        """No-op.  Returns zero-valued losses so runner's logging code stays happy."""
        zero = torch.tensor(0.0, device=self.device)
        # mimic MADDPGController.update: returns ([critic_loss_uav, critic_loss_charger], policy_loss_or_None)
        # runner.py expects the result to be tuple(list, list_or_None)
        return [zero, zero], [zero, zero]

    def update_target_net(self) -> None:
        return None

    def save_checkpoint(self, step_num: int, episode_num: int) -> None:
        """No parameters to save.  We still touch the checkpoint dir to leave a marker."""
        return None
