"""A per-player view of the game state.

The acting player can see everyone else's actual cards but not their own; for
their own hand they only have the *knowledge* derived from clues. This module
also provides small inference helpers that strategies commonly need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .actions import ActionRecord
from .cards import Card, Color


@dataclass(frozen=True)
class CardView:
    """What an observer knows about one card in some hand.

    ``card`` is ``None`` when the card belongs to the observer (hidden to
    themselves). ``possible_colors`` / ``possible_ranks`` are the candidate
    values consistent with the *direct clue information* received so far
    (positive and negative). ``clued`` is True once any clue has touched it.
    ``order`` is a stable unique id assigned when the card was drawn.
    """

    order: int
    card: Optional[Card]
    possible_colors: frozenset[Color]
    possible_ranks: frozenset[int]
    clued: bool


@dataclass(frozen=True)
class Observation:
    num_players: int
    player_index: int          # the observer (whose hand is hidden)
    current_player: int        # whose turn it is
    hands: tuple[tuple[CardView, ...], ...]
    play_stacks: dict[Color, int]
    discard_pile: tuple[Card, ...]
    clue_tokens: int
    max_clue_tokens: int
    strikes: int
    max_strikes: int
    deck_size: int
    colors: tuple[Color, ...]
    score: int
    log: tuple[ActionRecord, ...]

    # --- basic accessors --------------------------------------------------
    @property
    def own_hand(self) -> tuple[CardView, ...]:
        return self.hands[self.player_index]

    def other_players(self) -> list[int]:
        return [p for p in range(self.num_players) if p != self.player_index]

    # --- playability / usefulness inference -------------------------------
    def next_rank(self, color: Color) -> int:
        """The rank that would currently play on ``color`` (1..5, or 6 if done)."""
        return self.play_stacks[color] + 1

    def is_playable(self, card: Card) -> bool:
        return card.rank == self.play_stacks[card.color] + 1

    def is_dead(self, card: Card) -> bool:
        """True if the card can never be played (its stack is already past it)."""
        return card.rank <= self.play_stacks[card.color]

    def known_playable(self, hand_index: int) -> bool:
        """True if *every* value still possible for this own-hand card plays now.

        Conservative: uses only direct clue knowledge (marginal color/rank
        candidates), so it returns True only when playability is certain.
        """
        view = self.own_hand[hand_index]
        for color in view.possible_colors:
            for rank in view.possible_ranks:
                if rank != self.play_stacks[color] + 1:
                    return False
        return bool(view.possible_colors) and bool(view.possible_ranks)

    def known_dead(self, hand_index: int) -> bool:
        """True if every value still possible for this own-hand card is unplayable forever."""
        view = self.own_hand[hand_index]
        for color in view.possible_colors:
            for rank in view.possible_ranks:
                if rank > self.play_stacks[color]:
                    return False
        return bool(view.possible_colors) and bool(view.possible_ranks)
