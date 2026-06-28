"""Action types players can take, plus a record type for the game log."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .cards import Card, Color


class ActionType(Enum):
    PLAY = "play"
    DISCARD = "discard"
    CLUE_COLOR = "clue_color"
    CLUE_RANK = "clue_rank"


@dataclass(frozen=True)
class Action:
    """A single move.

    - PLAY / DISCARD use ``card_index`` (index into the *actor's* own hand).
    - CLUE_COLOR uses ``target`` + ``color``.
    - CLUE_RANK uses ``target`` + ``rank``.
    """

    type: ActionType
    card_index: Optional[int] = None
    target: Optional[int] = None
    color: Optional[Color] = None
    rank: Optional[int] = None

    # --- convenience constructors -----------------------------------------
    @staticmethod
    def play(card_index: int) -> "Action":
        return Action(ActionType.PLAY, card_index=card_index)

    @staticmethod
    def discard(card_index: int) -> "Action":
        return Action(ActionType.DISCARD, card_index=card_index)

    @staticmethod
    def clue_color(target: int, color: Color) -> "Action":
        return Action(ActionType.CLUE_COLOR, target=target, color=color)

    @staticmethod
    def clue_rank(target: int, rank: int) -> "Action":
        return Action(ActionType.CLUE_RANK, target=target, rank=rank)

    @property
    def is_clue(self) -> bool:
        return self.type in (ActionType.CLUE_COLOR, ActionType.CLUE_RANK)

    def __str__(self) -> str:
        if self.type is ActionType.PLAY:
            return f"PLAY slot {self.card_index}"
        if self.type is ActionType.DISCARD:
            return f"DISCARD slot {self.card_index}"
        if self.type is ActionType.CLUE_COLOR:
            return f"CLUE p{self.target} color {self.color}"
        return f"CLUE p{self.target} rank {self.rank}"


@dataclass
class ActionRecord:
    """An entry in the game log describing what happened on a turn."""

    turn: int
    player: int
    action: Action
    # Outcome details (populated by the engine):
    played_card: Optional[Card] = None
    success: Optional[bool] = None  # for PLAY: True=played, False=misplay
    discarded_card: Optional[Card] = None
    touched_orders: tuple[int, ...] = ()  # card ids touched by a clue
    drew_order: Optional[int] = None  # id of the replacement card drawn, if any
    acted_order: Optional[int] = None  # id of the card played/discarded

    def __str__(self) -> str:
        base = f"t{self.turn} p{self.player}: {self.action}"
        if self.action.type is ActionType.PLAY:
            tag = "OK" if self.success else "STRIKE"
            return f"{base} -> {self.played_card} [{tag}]"
        if self.action.type is ActionType.DISCARD:
            return f"{base} -> {self.discarded_card}"
        return f"{base} -> touched {len(self.touched_orders)}"
