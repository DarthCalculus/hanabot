"""Uniform-random legal-move player.

Primarily a baseline and an engine fuzzer: across many games it exercises every
code path in the rules engine. It needs the legal-action list, which it gets via
a small callback so strategies stay decoupled from ``GameState``.
"""

from __future__ import annotations

import random
from typing import Callable

from ..actions import Action
from ..observation import Observation
from .base import Player

LegalActionsFn = Callable[[Observation], list[Action]]


class RandomPlayer(Player):
    name = "random"

    def __init__(self, legal_actions_fn: LegalActionsFn, seed: int | None = None):
        self._legal = legal_actions_fn
        self.rng = random.Random(seed)

    def act(self, obs: Observation) -> Action:
        actions = self._legal(obs)
        return self.rng.choice(actions)
