"""Strategy interface.

A strategy is anything implementing :meth:`Player.act`, which maps an
:class:`~hanabi_sim.observation.Observation` to a legal
:class:`~hanabi_sim.actions.Action`. Strategies are pure decision functions:
they receive a fresh observation each turn and must not mutate engine state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..actions import Action
from ..observation import Observation


class Player(ABC):
    #: Human-readable strategy name (used in result summaries).
    name: str = "player"

    @abstractmethod
    def act(self, obs: Observation) -> Action:
        """Return the action to take given the current observation."""

    def reset(self) -> None:
        """Hook called at the start of each new game. Override if stateful."""
