"""Clue-target priority on top of goodtouch: clue the next player who has nothing
to play.

When choosing among play clues, *which player* to clue is decided before which
card (2/3/4) the focus is. A teammate who already has a globally-known card to
play -- one called by a clue, or known-playable to them -- will act on their own;
the teammate with no such card is the one who will otherwise be forced to
discard. So we prioritize the soonest player with no queued play, and only fall
back to players who already have one.
"""

from __future__ import annotations

from ..observation import Observation
from .good_touch_player import GoodTouchPlayer


class TempoPlayer(GoodTouchPlayer):
    name = "tempo"

    def _clue_target_key(self, obs: Observation, p: int) -> tuple:
        # 0 = player has nothing globally-known to play (prefer), 1 = already busy.
        has_play = 1 if self._player_has_play(obs, p) else 0
        dist = (p - obs.current_player) % obs.num_players
        return (has_play, dist)
