"""The Hanabi rules engine.

``GameState`` holds the full (god's-eye) state and applies actions according to
standard Hanabi rules. Strategies never touch this directly; they receive an
:class:`~hanabi_sim.observation.Observation` and return an
:class:`~hanabi_sim.actions.Action`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .actions import Action, ActionRecord, ActionType
from .cards import Card, Color, RANKS, STANDARD_COLORS, build_deck
from .observation import CardView, Observation

# Hand size by player count (standard Hanabi).
HAND_SIZE = {2: 5, 3: 5, 4: 4, 5: 4}

MAX_CLUE_TOKENS = 8
MAX_STRIKES = 3


@dataclass
class GameConfig:
    num_players: int
    colors: tuple[Color, ...] = STANDARD_COLORS
    max_clue_tokens: int = MAX_CLUE_TOKENS
    max_strikes: int = MAX_STRIKES
    #: If True (standard rules), reaching ``max_strikes`` zeroes the final score.
    loss_score_on_strikeout: bool = True

    def __post_init__(self) -> None:
        if not 2 <= self.num_players <= 5:
            raise ValueError("Hanabi supports 2-5 players")

    @property
    def hand_size(self) -> int:
        return HAND_SIZE[self.num_players]

    @property
    def max_score(self) -> int:
        return len(self.colors) * 5


class _HeldCard:
    """Engine-internal: a card in a hand plus the knowledge clued about it."""

    __slots__ = ("order", "card", "possible_colors", "possible_ranks", "clued")

    def __init__(self, order: int, card: Card, colors: tuple[Color, ...]):
        self.order = order
        self.card = card
        self.possible_colors: set[Color] = set(colors)
        self.possible_ranks: set[int] = set(RANKS)
        self.clued = False

    def view(self, hidden: bool) -> CardView:
        return CardView(
            order=self.order,
            card=None if hidden else self.card,
            possible_colors=frozenset(self.possible_colors),
            possible_ranks=frozenset(self.possible_ranks),
            clued=self.clued,
        )


@dataclass
class GameResult:
    score: int            # realized final score (0 on strikeout under standard rules)
    stack_total: int      # sum of stacks regardless of strikeout
    won: bool             # perfect game (== max_score)
    strikeout: bool       # ended via reaching max_strikes
    strikes: int
    turns: int
    clue_tokens_left: int


class IllegalAction(Exception):
    pass


class GameState:
    def __init__(self, config: GameConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = random.Random(seed)
        self.num_players = config.num_players
        self.colors = config.colors

        self._next_order = 0
        self.deck: list[Card] = build_deck(self.colors)
        self.rng.shuffle(self.deck)

        self.hands: list[list[_HeldCard]] = [[] for _ in range(self.num_players)]
        self.play_stacks: dict[Color, int] = {c: 0 for c in self.colors}
        self.discard_pile: list[Card] = []
        self.clue_tokens = config.max_clue_tokens
        self.strikes = 0

        self.current_player = 0
        self.turn_count = 0
        self.final_turns_remaining: Optional[int] = None
        self.game_over = False
        self.strikeout = False
        self.log: list[ActionRecord] = []

        self._deal()

    # --- setup ------------------------------------------------------------
    def _deal(self) -> None:
        for _ in range(self.config.hand_size):
            for p in range(self.num_players):
                self._draw_into(self.hands[p])

    def _draw_into(self, hand: list[_HeldCard]) -> Optional[int]:
        if not self.deck:
            return None
        card = self.deck.pop()
        held = _HeldCard(self._next_order, card, self.colors)
        self._next_order += 1
        hand.append(held)
        return held.order

    # --- queries ----------------------------------------------------------
    @property
    def score(self) -> int:
        return sum(self.play_stacks.values())

    @property
    def deck_size(self) -> int:
        return len(self.deck)

    def legal_actions(self, player: Optional[int] = None) -> list[Action]:
        """All legal actions for ``player`` (defaults to the current player)."""
        if player is None:
            player = self.current_player
        actions: list[Action] = []
        hand = self.hands[player]

        # Play / discard any held card.
        for i in range(len(hand)):
            actions.append(Action.play(i))
        if self.clue_tokens < self.config.max_clue_tokens:
            for i in range(len(hand)):
                actions.append(Action.discard(i))

        # Clues: only if a token is available and the clue touches >=1 card.
        if self.clue_tokens > 0:
            for target in range(self.num_players):
                if target == player:
                    continue
                thand = self.hands[target]
                colors_present = {hc.card.color for hc in thand}
                ranks_present = {hc.card.rank for hc in thand}
                for color in self.colors:
                    if color in colors_present:
                        actions.append(Action.clue_color(target, color))
                for rank in RANKS:
                    if rank in ranks_present:
                        actions.append(Action.clue_rank(target, rank))
        return actions

    def observation(self, player: Optional[int] = None) -> Observation:
        if player is None:
            player = self.current_player
        hands_view = tuple(
            tuple(hc.view(hidden=(p == player)) for hc in self.hands[p])
            for p in range(self.num_players)
        )
        return Observation(
            num_players=self.num_players,
            player_index=player,
            current_player=self.current_player,
            hands=hands_view,
            play_stacks=dict(self.play_stacks),
            discard_pile=tuple(self.discard_pile),
            clue_tokens=self.clue_tokens,
            max_clue_tokens=self.config.max_clue_tokens,
            strikes=self.strikes,
            max_strikes=self.config.max_strikes,
            deck_size=len(self.deck),
            colors=self.colors,
            score=self.score,
            log=tuple(self.log),
        )

    # --- mutation ---------------------------------------------------------
    def apply(self, action: Action) -> ActionRecord:
        if self.game_over:
            raise IllegalAction("game is already over")
        actor = self.current_player
        hand = self.hands[actor]
        record = ActionRecord(turn=self.turn_count, player=actor, action=action)

        if action.type is ActionType.PLAY:
            self._apply_play(hand, action, record)
        elif action.type is ActionType.DISCARD:
            self._apply_discard(hand, action, record)
        elif action.type in (ActionType.CLUE_COLOR, ActionType.CLUE_RANK):
            self._apply_clue(actor, action, record)
        else:  # pragma: no cover - defensive
            raise IllegalAction(f"unknown action type: {action.type}")

        self.log.append(record)

        # Terminal conditions checked before advancing the turn.
        if self.strikes >= self.config.max_strikes:
            self.game_over = True
            self.strikeout = True
            return record
        if self.score == self.config.max_score:
            self.game_over = True
            return record

        # Endgame: once the last card is drawn, every player gets exactly one
        # more turn -- INCLUDING the player who drew it (so they can still play
        # a just-drawn card, e.g. the final 5). That is a full lap of
        # `num_players` turns *after* the drawing turn.
        #
        # The drawing turn must not be counted as one of those, so on a single
        # turn we either start the countdown OR decrement it, never both: the
        # turn that empties the deck only arms the counter; later turns spend it.
        if self.final_turns_remaining is not None:
            self.final_turns_remaining -= 1
            if self.final_turns_remaining <= 0:
                self.game_over = True
        elif not self.deck:
            self.final_turns_remaining = self.num_players

        self.current_player = (self.current_player + 1) % self.num_players
        self.turn_count += 1
        return record

    def _apply_play(self, hand, action, record) -> None:
        i = action.card_index
        if i is None or not (0 <= i < len(hand)):
            raise IllegalAction(f"bad play index {i}")
        held = hand.pop(i)
        card = held.card
        record.played_card = card
        record.acted_order = held.order
        if card.rank == self.play_stacks[card.color] + 1:
            self.play_stacks[card.color] = card.rank
            record.success = True
            # Completing a stack (playing a 5) returns a clue token.
            if card.rank == 5 and self.clue_tokens < self.config.max_clue_tokens:
                self.clue_tokens += 1
        else:
            record.success = False
            self.strikes += 1
            self.discard_pile.append(card)
        record.drew_order = self._draw_into(hand)

    def _apply_discard(self, hand, action, record) -> None:
        if self.clue_tokens >= self.config.max_clue_tokens:
            raise IllegalAction("cannot discard at max clue tokens")
        i = action.card_index
        if i is None or not (0 <= i < len(hand)):
            raise IllegalAction(f"bad discard index {i}")
        held = hand.pop(i)
        record.discarded_card = held.card
        record.acted_order = held.order
        self.discard_pile.append(held.card)
        self.clue_tokens += 1
        record.drew_order = self._draw_into(hand)

    def _apply_clue(self, actor, action, record) -> None:
        if self.clue_tokens <= 0:
            raise IllegalAction("no clue tokens available")
        target = action.target
        if target is None or not (0 <= target < self.num_players):
            raise IllegalAction(f"bad clue target {target}")
        if target == actor:
            raise IllegalAction("cannot clue yourself")
        thand = self.hands[target]

        touched: list[int] = []
        if action.type is ActionType.CLUE_COLOR:
            color = action.color
            for hc in thand:
                if hc.card.color == color:
                    hc.possible_colors = {color}
                    hc.clued = True
                    touched.append(hc.order)
                else:
                    hc.possible_colors.discard(color)
        else:  # CLUE_RANK
            rank = action.rank
            for hc in thand:
                if hc.card.rank == rank:
                    hc.possible_ranks = {rank}
                    hc.clued = True
                    touched.append(hc.order)
                else:
                    hc.possible_ranks.discard(rank)

        if not touched:
            raise IllegalAction("clue must touch at least one card")
        self.clue_tokens -= 1
        record.touched_orders = tuple(touched)

    # --- result -----------------------------------------------------------
    def result(self) -> GameResult:
        stack_total = self.score
        if self.strikeout and self.config.loss_score_on_strikeout:
            final = 0
        else:
            final = stack_total
        return GameResult(
            score=final,
            stack_total=stack_total,
            won=(stack_total == self.config.max_score),
            strikeout=self.strikeout,
            strikes=self.strikes,
            turns=self.turn_count,
            clue_tokens_left=self.clue_tokens,
        )
