"""Deduce the color of a known 5 by card-counting, on top of :class:`FocusPlayer`.

A 5 is unique -- one per color -- so a player holding a card known to be a 5 can
often pin down its color without being told:

* every color whose 5 is visible *elsewhere* (a teammate's hand, the played
  stacks, or the discard pile) cannot be this card, so it is ruled out;
* every color ruled out by a color clue (the 5 was in hand when a clue of that
  color missed it) is already gone from the card's candidate colors -- the
  engine records that negative information automatically.

So the candidate colors are ``possible_colors`` minus the colors seen elsewhere.
When exactly one survives, the 5's color is known (its true color can never be
ruled out, since the only copy is the card itself), and it is played as soon as
that color's stack reaches 4.
"""

from __future__ import annotations

from ..actions import Action
from ..observation import CardView, Observation
from .focus_player import FocusPlayer


class DeduceFivePlayer(FocusPlayer):
    name = "deduce5"

    def _fives_seen_elsewhere(self, obs: Observation) -> set:
        seen = set()
        for p in obs.other_players():
            for cv in obs.hands[p]:
                if cv.card.rank == 5:
                    seen.add(cv.card.color)
        for c in obs.colors:
            if obs.play_stacks[c] == 5:  # that 5 is on the board
                seen.add(c)
        for card in obs.discard_pile:
            if card.rank == 5:
                seen.add(card.color)
        return seen

    def _deduced_five_color(self, obs: Observation, view: CardView):
        """The color of ``view`` if it is a known 5 narrowed to one color, else None."""
        if view.possible_ranks != {5}:
            return None
        seen = self._fives_seen_elsewhere(obs)
        candidates = [c for c in view.possible_colors if c not in seen]
        return candidates[0] if len(candidates) == 1 else None

    def _choose_play(self, obs: Observation) -> Action | None:
        # Inherited plays first (known/called cards, lowest rank).
        action = super()._choose_play(obs)
        if action is not None:
            return action
        # Then a 5 whose color we've deduced, once that stack is ready.
        for i, view in enumerate(obs.own_hand):
            color = self._deduced_five_color(obs, view)
            if color is not None and obs.play_stacks[color] == 4:
                return Action.play(i)
        return None
