"""Adds the *multiple-1s* convention on top of :class:`FiveSavePlayer`.

A single rank-1 clue can launch several plays at once, so it's the most
efficient opener. This strategy:

* **Interprets** a rank-1 clue (any number of cards) as "play *all* of these" --
  every touched 1 becomes a play call (extending the singleton-only rule).
* **Gives** a rank-1 clue preferentially when a teammate has two or more
  playable 1s -- but only when doing so is unambiguous and non-wasteful:

  - never reclue a card already known to be a 1,
  - never clue a hand where rank-1 would touch a *dead* 1 (its color is played),
  - never clue same-color 1s in one hand (a color-blind receiver would play the
    duplicate and misplay),
  - never clue a 1 whose color is already "spoken for" -- played, or a 1 of that
    color already called to play somewhere (a duplicate that will be played).

Single playable 1s are still handled by the inherited singleton play-clue logic.
"""

from __future__ import annotations

from ..actions import Action, ActionType
from ..observation import Observation
from .five_save_player import FiveSavePlayer


class OnesPlayer(FiveSavePlayer):
    name = "ones"

    # --- receiving: a rank-1 clue calls every 1 it touches ----------------
    def _clue_play_targets(self, rec, stacks) -> tuple:
        a = rec.action
        if a.type is ActionType.CLUE_RANK and a.rank == 1:
            return rec.touched_orders
        return super()._clue_play_targets(rec, stacks)

    # --- giving: prefer the multi-1 clue ----------------------------------
    def _choose_play_clue(self, obs: Observation) -> Action | None:
        return self._ones_clue(obs) or super()._choose_play_clue(obs)

    def _spoken_for_one_colors(self, obs: Observation) -> set:
        """Colors whose 1 is already played, or already going to be played (called
        by a clue or known-playable to its holder)."""
        colors = {c for c in obs.colors if obs.play_stacks[c] >= 1}
        for p in range(obs.num_players):
            called_p = self._derive_called(obs, p)
            for cv in obs.hands[p]:
                if cv.card is None or cv.card.rank != 1:
                    continue
                if cv.order in called_p or self._view_known_playable(cv, obs.play_stacks):
                    colors.add(cv.card.color)
        return colors

    def _ones_clue(self, obs: Observation) -> Action | None:
        spoken = self._spoken_for_one_colors(obs)
        best: tuple[int, Action] | None = None
        for p in obs.other_players():
            # A rank-1 clue touches exactly the 1s in the hand.
            ones = [cv for cv in obs.hands[p] if cv.card.rank == 1]
            if len(ones) < 2:
                continue  # singleton 1s are handled by the base play-clue logic
            if any(cv.possible_ranks == {1} for cv in ones):
                continue  # would reclue an already-known 1
            colors = [cv.card.color for cv in ones]
            if any(obs.play_stacks[c] >= 1 for c in colors):
                continue  # a touched 1 is already dead -> would misplay
            if len(set(colors)) != len(colors):
                continue  # same-color 1s in hand -> duplicate misplay
            if any(c in spoken for c in colors):
                continue  # a touched 1 duplicates one that will be played
            dist = (p - obs.current_player) % obs.num_players
            if best is None or dist < best[0]:
                best = (dist, Action.clue_rank(p, 1))
        return best[1] if best else None
