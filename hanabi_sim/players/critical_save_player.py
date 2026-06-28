"""Save a critical low card on a teammate's chop, on top of tempo.

A rank-r clue (r in {2,3,4}) given while rank r is NOT currently playable (no
stack at r-1) cannot be read as a play -- the receiver's consistent-playable test
fails -- so it acts as a SAVE: the touched card becomes clued (hence protected
from discard) and is played later once its color is pinned, just like a saved 5.
We use it to protect the last surviving copy of a still-needed 2/3/4 that's about
to be discarded off a teammate's chop.

The receiver needs no new logic: such a clue is already not called as a play
(consistency fails), and a clued card is already off the chop.

Limitation: only works when the card's rank isn't playable elsewhere. If it is, a
rank clue would mean "play" and a color clue would leave the rank open (the
receiver would bomb it), so a single clue can't save it.
"""

from __future__ import annotations

from ..actions import Action
from ..cards import RANK_COUNTS
from ..observation import Observation
from .tempo_player import TempoPlayer


class CriticalSavePlayer(TempoPlayer):
    name = "critsave"

    def _urgent_save(self, obs: Observation):
        return self._five_save(obs) or self._critical_save(obs)

    def _critical_save(self, obs: Observation) -> Action | None:
        nxt = (obs.player_index + 1) % obs.num_players
        if nxt == obs.player_index:
            return None
        if self._player_has_play(obs, nxt):
            return None  # they have a play, so won't discard this turn
        chop = self._chop_index(obs, nxt)
        if chop is None:
            return None
        card = obs.hands[nxt][chop].card
        if card.rank not in (2, 3, 4):
            return None  # 5s are handled by _five_save
        if self._rank_playable(obs, card.rank):
            return None  # rank is playable -> a rank clue would read as a play
        if not self._is_critical(obs, card):
            return None  # only the last surviving copy is worth a save
        return Action.clue_rank(nxt, card.rank)

    def _rank_playable(self, obs: Observation, r: int) -> bool:
        return any(obs.play_stacks[c] == r - 1 for c in obs.colors)

    def _is_critical(self, obs: Observation, card) -> bool:
        if card.rank <= obs.play_stacks[card.color]:
            return False  # already played -> not needed
        discarded = sum(1 for c in obs.discard_pile if c == card)
        return discarded >= RANK_COUNTS[card.rank] - 1  # last surviving copy
