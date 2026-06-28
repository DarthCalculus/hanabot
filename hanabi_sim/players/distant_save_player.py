"""Save critical cards on non-immediately-next players too, on top of critsave.

``critsave`` only saves the *immediate* next player's chop (urgently, before our
own play), on the theory that later players get saved by whoever sits before
them. This strategy adds a lower-priority save: when we'd otherwise just discard,
we instead protect a critical card sitting on a *further-out* teammate's chop.

Priority: urgent next-player save -> play -> play clue -> **distant save** ->
trash-1 clue -> discard. So it never displaces a play or a normal clue; it only
replaces a would-be discard with something useful.

Uses the same save signal as critsave (a rank clue while that rank isn't
playable, so it reads as a save, not a play) and only fires for a last-copy
2/3/4 on a teammate who currently has nothing to play.
"""

from __future__ import annotations

from ..actions import Action
from ..observation import Observation
from .critical_save_player import CriticalSavePlayer


class DistantSavePlayer(CriticalSavePlayer):
    name = "distsave"

    def _save_clue_for(self, obs: Observation, p: int) -> Action | None:
        """A critsave-style save of player ``p``'s chop, or None."""
        if self._player_has_play(obs, p):
            return None  # they have a play, won't discard their chop soon
        chop = self._chop_index(obs, p)
        if chop is None:
            return None
        card = obs.hands[p][chop].card
        if card.rank not in (2, 3, 4) or self._rank_playable(obs, card.rank):
            return None  # not a 2/3/4, or the rank is playable (would read as a play)
        if not self._is_critical(obs, card):
            return None  # only the last surviving copy is worth a save
        return Action.clue_rank(p, card.rank)

    def _distant_save(self, obs: Observation) -> Action | None:
        # Players that don't come immediately next, soonest first.
        for dist in range(2, obs.num_players):
            clue = self._save_clue_for(obs, (obs.player_index + dist) % obs.num_players)
            if clue is not None:
                return clue
        return None

    def _discard_one_clue(self, obs: Observation) -> Action | None:
        # Prefer protecting a distant teammate's critical chop over discarding
        # (and over the trash-1 clue).
        dsave = self._distant_save(obs)
        if dsave is not None:
            return dsave
        return super()._discard_one_clue(obs)
