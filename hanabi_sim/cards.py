"""Card model and deck construction for Hanabi.

A standard Hanabi deck has 5 suits (colors). Each suit contains the ranks
1..5 with multiplicities [1->3, 2->2, 3->2, 4->2, 5->1], i.e. 10 cards per
suit and 50 cards total.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Color(Enum):
    """The five standard (no-variant) Hanabi suits."""

    RED = "R"
    YELLOW = "Y"
    GREEN = "G"
    BLUE = "B"
    PURPLE = "P"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Color.{self.name}"


#: Ranks present in every suit, in ascending play order.
RANKS: tuple[int, ...] = (1, 2, 3, 4, 5)

#: How many copies of each rank appear in a suit.
RANK_COUNTS: dict[int, int] = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}

#: The five standard colors, in a canonical order.
STANDARD_COLORS: tuple[Color, ...] = (
    Color.RED,
    Color.YELLOW,
    Color.GREEN,
    Color.BLUE,
    Color.PURPLE,
)


@dataclass(frozen=True)
class Card:
    """An immutable Hanabi card identified by its color and rank."""

    color: Color
    rank: int

    def __post_init__(self) -> None:
        if self.rank not in RANK_COUNTS:
            raise ValueError(f"invalid rank: {self.rank!r}")

    def __str__(self) -> str:
        return f"{self.color.value}{self.rank}"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Card({self.color.name}, {self.rank})"


def build_deck(colors: tuple[Color, ...] = STANDARD_COLORS) -> list[Card]:
    """Return one full, *unshuffled* Hanabi deck for the given colors."""
    deck: list[Card] = []
    for color in colors:
        for rank, count in RANK_COUNTS.items():
            deck.extend(Card(color, rank) for _ in range(count))
    return deck
