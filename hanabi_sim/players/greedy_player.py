"""A simple, transparent heuristic baseline.

This is intentionally *not* a convention-based bot. It reasons only from direct
clue knowledge (no implicit "this clue means play it" signalling), which makes
it a clean, predictable yardstick to measure smarter strategies against.

Decision order each turn:
  1. Play a card that is *known* to be playable (certainty from clues).
  2. Otherwise, if a clue token is available, give the clue that reveals the
     most fresh, currently-playable cards in a teammate's hand.
  3. Otherwise discard: prefer a card known to be dead, else the "chop"
     (oldest unclued card).
  4. If forced (at max tokens with nothing better), give any legal clue.
"""

from __future__ import annotations

from ..actions import Action
from ..cards import RANKS, Card
from ..observation import CardView, Observation
from .base import Player


class GreedyPlayer(Player):
    name = "greedy"

    def act(self, obs: Observation) -> Action:
        play = self._best_play(obs)
        if play is not None:
            return play

        can_discard = obs.clue_tokens < obs.max_clue_tokens

        if obs.clue_tokens > 0:
            clue = self._best_clue(obs)
            if clue is not None:
                return clue

        if can_discard:
            return self._best_discard(obs)

        # Forced: at max tokens, no play, no useful clue -> must give some clue.
        return self._any_clue(obs)

    # --- play -------------------------------------------------------------
    def _best_play(self, obs: Observation) -> Action | None:
        best_index = None
        best_rank = 99
        for i in range(len(obs.own_hand)):
            if obs.known_playable(i):
                min_rank = min(obs.own_hand[i].possible_ranks)
                if min_rank < best_rank:
                    best_rank = min_rank
                    best_index = i
        return None if best_index is None else Action.play(best_index)

    # --- clues ------------------------------------------------------------
    def _clue_candidates(self, obs: Observation):
        """Yield (action, touched_views) for every legal clue this player can give."""
        for target in obs.other_players():
            hand = obs.hands[target]
            colors_present = {cv.card.color for cv in hand}
            ranks_present = {cv.card.rank for cv in hand}
            for color in obs.colors:
                if color in colors_present:
                    touched = [cv for cv in hand if cv.card.color == color]
                    yield Action.clue_color(target, color), touched
            for rank in RANKS:
                if rank in ranks_present:
                    touched = [cv for cv in hand if cv.card.rank == rank]
                    yield Action.clue_rank(target, rank), touched

    def _best_clue(self, obs: Observation) -> Action | None:
        best = None
        best_key = (0, 0)  # (fresh playable touched, -total touched)
        for action, touched in self._clue_candidates(obs):
            fresh_playable = sum(
                1 for cv in touched if not cv.clued and obs.is_playable(cv.card)
            )
            if fresh_playable == 0:
                continue
            key = (fresh_playable, -len(touched))
            if key > best_key:
                best_key = key
                best = action
        return best

    def _any_clue(self, obs: Observation) -> Action:
        # First legal clue available (hands are never empty during play).
        for action, _ in self._clue_candidates(obs):
            return action
        raise RuntimeError("no legal clue available")  # pragma: no cover

    # --- discard ----------------------------------------------------------
    def _best_discard(self, obs: Observation) -> Action:
        hand = obs.own_hand
        # 1) A card known to be dead is always safe to dump.
        for i in range(len(hand)):
            if obs.known_dead(i):
                return Action.discard(i)
        # 2) Chop = oldest unclued card (index 0 is the oldest).
        for i in range(len(hand)):
            if not hand[i].clued:
                return Action.discard(i)
        # 3) Everything is clued; discard the oldest.
        return Action.discard(0)
