"""Adds a *trash-1 discard clue* on top of :class:`OnesPlayer`.

Dead 1s (a 1 whose color is already started) are pure trash. If a teammate is
holding several of them, we can tell them -- with a rank-1 clue -- to discard
those instead of blindly discarding an unknown card off their chop (which might
be a card the team still needs). That routes discards onto known trash and
protects critical cards.

Disambiguating a discard clue from a play clue (both are rank-1):

    open = number of colors whose stack is still 0
    a rank-1 clue touching N cards is a PLAY clue if N <= open
                                       a DISCARD clue if N >  open

If N > open it's impossible for all touched 1s to be distinct *playable* 1s, so
there's no risk of reading it as a play clue -- the receiver concludes they are
trash. The giver only ever gives a rank-1 clue that is cleanly one or the other
(its play-clue path already refuses mixed/duplicate hands), and only gives the
discard clue when *every* touched 1 is in fact dead, so the two never collide.

Because a rank-1 clue now always carries a signal (play or discard), it is never
used as an innocent stall -- handled generically by ``_clue_action_targets``.
"""

from __future__ import annotations

from ..actions import Action, ActionType
from ..observation import Observation
from .ones_player import OnesPlayer


def _open_slots(stacks) -> int:
    return sum(1 for v in stacks.values() if v == 0)


class OnesDiscardPlayer(OnesPlayer):
    name = "onesdisc"

    # --- receiving: split rank-1 clues into play vs discard ---------------
    def _clue_play_targets(self, rec, stacks) -> tuple:
        a = rec.action
        if a.type is ActionType.CLUE_RANK and a.rank == 1:
            # Too many 1s to all be playable -> it's a discard clue, not a play.
            if len(rec.touched_orders) > _open_slots(stacks):
                return ()
            return rec.touched_orders
        return super()._clue_play_targets(rec, stacks)

    def _clue_discard_targets(self, rec, stacks) -> tuple:
        a = rec.action
        if a.type is ActionType.CLUE_RANK and a.rank == 1:
            if len(rec.touched_orders) > _open_slots(stacks):
                return rec.touched_orders
        return ()

    # --- giving: trash-1 discard clue, as an alternative to discarding ----
    def act(self, obs: Observation) -> Action:
        if obs.clue_tokens > 0:
            save = self._urgent_save(obs)
            if save is not None:
                return save
        play = self._choose_play(obs)
        if play is not None:
            return play
        if obs.clue_tokens > 0:
            clue = self._choose_play_clue(obs)
            if clue is not None:
                return clue
        # Rather than let a teammate blindly discard, tell them to dump trash 1s.
        if obs.clue_tokens > 0:
            disc = self._discard_one_clue(obs)
            if disc is not None:
                return disc
        if obs.clue_tokens < obs.max_clue_tokens:
            return self._choose_discard(obs)
        return self._stall_clue(obs)

    def _urgent_save(self, obs: Observation):
        """Urgent clue to stop the next player discarding something important.
        Subclasses extend (e.g. to also save critical low cards)."""
        return self._five_save(obs)

    @staticmethod
    def _view_known_dead(view, stacks) -> bool:
        """True if every value still possible for ``view`` is already unplayable."""
        if not view.possible_colors or not view.possible_ranks:
            return False
        return all(
            r <= stacks[c] for c in view.possible_colors for r in view.possible_ranks
        )

    def _would_discard_random(self, obs: Observation, p: int) -> bool:
        """(A) Player ``p`` has no safe discard, so their next discard is random."""
        trash = self._derive_trash(obs, p)
        for cv in obs.hands[p]:
            if cv.order in trash or self._view_known_dead(cv, obs.play_stacks):
                return False
        return True

    def _discard_one_clue(self, obs: Observation) -> Action | None:
        open_slots = _open_slots(obs.play_stacks)
        best: tuple[int, Action] | None = None
        for p in obs.other_players():
            ones = [cv for cv in obs.hands[p] if cv.card.rank == 1]
            if len(ones) < 2:
                continue  # (B) clue at least two 1s at once
            # Every touched 1 must be dead (its color already started)...
            if not all(obs.play_stacks[cv.card.color] >= 1 for cv in ones):
                continue
            # ...and there must be more 1s than could be played, so it reads as
            # a discard clue rather than a play clue.
            if len(ones) <= open_slots:
                continue
            # (A) only worth it if they'd otherwise discard a random card.
            if not self._would_discard_random(obs, p):
                continue
            trash = self._derive_trash(obs, p)
            if all(cv.order in trash for cv in ones):
                continue  # they already know these are trash
            dist = (p - obs.current_player) % obs.num_players
            if best is None or dist < best[0]:
                best = (dist, Action.clue_rank(p, 1))
        return best[1] if best else None

    # --- discarding: dump known trash first -------------------------------
    def _choose_discard(self, obs: Observation) -> Action:
        trash = self._derive_trash(obs, obs.player_index)
        for i, cv in enumerate(obs.own_hand):
            if cv.order in trash:
                return Action.discard(i)
        return super()._choose_discard(obs)
