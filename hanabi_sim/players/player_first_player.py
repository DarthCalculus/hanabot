"""Player choice takes precedence over good-touch / clue-type when play-cluing.

Earlier strategies chose the *kind* of clue first (a clean multi-card clue, then
a single-card clue, then a bad-touch multi), with the player only a tie-break
inside each pass -- so a clean multi-card clue to a farther player would beat a
clean single-card clue to the nearer player. Here we rank every play-clue
candidate (multi and single together) with the PLAYER first:

    (player_has_a_play, distance, bad_touch, is_single, focus_rank)

So we clue the soonest player who has nothing to play, and only then prefer a
clean clue and a multi-card focus. (The multi-1 opener still goes first.)
"""

from __future__ import annotations

from ..actions import Action
from ..observation import Observation
from .chop_save_player import ChopSavePlayer


class PlayerFirstPlayer(ChopSavePlayer):
    name = "pfirst"

    def _choose_play_clue(self, obs: Observation) -> Action | None:
        ones = self._ones_clue(obs)
        if ones is not None:
            return ones

        will_orders, will_ids = self._will_play(obs)
        best_key = None
        best_act = None

        def consider(key, act):
            nonlocal best_key, best_act
            if best_key is None or key < best_key:
                best_key, best_act = key, act

        for p in obs.other_players():
            tgt = self._clue_target_key(obs, p)  # (has_play, distance)
            hand = obs.hands[p]
            colors = [cv.card.color for cv in hand]
            ranks = [cv.card.rank for cv in hand]

            # Multi-card play clues (color, or rank 2/3/4, touching >=2).
            options = [
                (Action.clue_color(p, c), [cv for cv in hand if cv.card.color == c])
                for c in set(colors)
            ] + [
                (Action.clue_rank(p, r), [cv for cv in hand if cv.card.rank == r])
                for r in (2, 3, 4)
            ]
            for act, touched in options:
                if len(touched) < 2:
                    continue
                focus = max(touched, key=lambda cv: cv.order)
                if not obs.is_playable(focus.card):
                    continue
                if focus.order in will_orders or (focus.card.color, focus.card.rank) in will_ids:
                    continue
                bad = 1 if self._bad_touch(obs, p, act, touched, focus) else 0
                consider(tgt + (bad, 0, focus.card.rank), act)

            # Single-card play clues (always clean -- only the focus is touched).
            for cv in hand:
                if not obs.is_playable(cv.card):
                    continue
                if cv.order in will_orders or (cv.card.color, cv.card.rank) in will_ids:
                    continue
                clue = self._play_clue_for(obs, p, cv, colors, ranks)
                if clue is not None:
                    consider(tgt + (0, 1, cv.card.rank), clue)

        return best_act
