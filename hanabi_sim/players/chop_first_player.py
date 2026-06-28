"""Like :class:`FiveSavePlayer`, but play clues are ranked tempo-first, using
chop-proximity as the tie-breaker.

Tempo comes first: clue the playable card held by the player who acts soonest,
so it gets played without delay. Among candidates that are equally soon, prefer
the one closest to chop -- it's the most at risk of being discarded, so cluing
it both rescues the card and signals the play. Rank breaks any remaining ties.

"Chop distance" = how many of that player's *unclued* cards are older than this
one. The chop (oldest unclued card) has distance 0; the next-oldest unclued card
has distance 1; and so on.
"""

from __future__ import annotations

from ..observation import CardView, Observation
from .five_save_player import FiveSavePlayer


class ChopFirstPlayer(FiveSavePlayer):
    name = "chopfirst"

    def _play_clue_priority(self, obs: Observation, p: int, idx: int, cv: CardView) -> tuple:
        dist = (p - obs.current_player) % obs.num_players
        # Cards in slots older than idx that are still unclued sit between this
        # card and the discard pile; fewer of them => closer to being discarded.
        chop_distance = sum(1 for older in obs.hands[p][:idx] if not older.clued)
        return (dist, chop_distance, cv.card.rank)
