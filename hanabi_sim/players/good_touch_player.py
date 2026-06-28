"""Good-touch refinement of multi-card play-clue priority, on top of deduce5.

A multi-card play clue fills in partial info on every card it touches, not just
the focus -- and a card with partial info is protected from being discarded.
That's fine when the extra cards are useful, or are trash the receiver can
*recognize* as trash (so they'll pitch them anyway). But if an extra card is
trash the receiver CANNOT yet deduce is trash, the clue quietly makes them hold
it and discard something better instead. In that case a single-card play clue is
preferable.

"Trash the receiver can't deduce" is judged against the play stacks as they would
stand after every card the giver can see queued to play (including the focus
we're about to clue) has played: an extra card is bad-touch if it is in fact dead
by then, yet its post-clue possibilities still include a card that could be alive
then.

Priority: multi-1 clue -> a *clean* multi-card play clue (no bad-touch) -> a
single-card play clue -> (last resort) a multi-card clue that does bad-touch,
since getting the play still beats not cluing.
"""

from __future__ import annotations

from ..actions import ActionType
from ..observation import Observation
from .deduce_five_player import DeduceFivePlayer
from .play_clue_player import PlayCluePlayer


class GoodTouchPlayer(DeduceFivePlayer):
    name = "goodtouch"

    def _choose_play_clue(self, obs: Observation) -> Action | None:
        ones = self._ones_clue(obs)
        if ones is not None:
            return ones
        clean = self._multi_play_clue(obs, reject=self._bad_touch)
        if clean is not None:
            return clean
        single = PlayCluePlayer._choose_play_clue(self, obs)
        if single is not None:
            return single
        # Last resort: a multi-card play clue even if it bad-touches -- a play now
        # beats no clue at all.
        return self._multi_play_clue(obs)

    # --- good touch -------------------------------------------------------
    def _hypo_stacks(self, obs: Observation, focus) -> dict:
        """Stacks after every visible queued play (and the focus) has resolved."""
        will_orders, _ = self._will_play(obs)
        queued: dict = {}
        for p in range(obs.num_players):
            for cv in obs.hands[p]:
                if cv.card is not None and cv.order in will_orders:
                    queued.setdefault(cv.card.color, set()).add(cv.card.rank)
        queued.setdefault(focus.card.color, set()).add(focus.card.rank)
        hypo = dict(obs.play_stacks)
        for c in obs.colors:
            ranks = queued.get(c, ())
            while hypo[c] + 1 in ranks:
                hypo[c] += 1
        return hypo

    def _bad_touch(self, obs, p, act, touched, focus) -> bool:
        """True if the clue touches a non-focus card that is trash yet the
        receiver couldn't deduce it's trash (given the clue + hypothetical stacks)."""
        hypo = self._hypo_stacks(obs, focus)
        for cv in touched:
            if cv.order == focus.order:
                continue  # the focus is the play target, not a bad touch
            card = cv.card
            if card.rank > hypo[card.color]:
                continue  # still genuinely useful -> fine to touch
            # It's trash. Would the receiver know that, post-clue, under hypo stacks?
            if act.type is ActionType.CLUE_RANK:
                colors, ranks = cv.possible_colors, (act.rank,)
            else:
                colors, ranks = (act.color,), cv.possible_ranks
            deducible = all(rk <= hypo[col] for col in colors for rk in ranks)
            if not deducible:
                return True  # trash that is hard to deduce as trash
        return False
