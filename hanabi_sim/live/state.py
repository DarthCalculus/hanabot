"""Adapter between hanab.live's game messages and our local engine types.

hanab.live identifies every card by a unique ``order`` (assigned in draw order),
suits by index (0=Red,1=Yellow,2=Green,3=Blue,4=Purple for "No Variant"), and
ranks 1..5 -- which line up exactly with our :mod:`hanabi_sim` representation. So
this tracker consumes the live action stream and produces an
:class:`~hanabi_sim.observation.Observation` (plus the action ``log`` the
strategies replay), letting any of our strategies play unchanged. It also
translates a chosen :class:`~hanabi_sim.actions.Action` back to a hanab.live
"action" message.

Only the no-variant 5-suit game is supported (our engine's assumption).
"""

from __future__ import annotations

from ..actions import Action, ActionRecord, ActionType
from ..cards import RANKS, STANDARD_COLORS, Card
from ..observation import CardView, Observation

SUITS = STANDARD_COLORS  # suit index i -> SUITS[i]
DECK_SIZE = 50


class LiveGameState:
    def __init__(self, our_player_index: int, num_players: int):
        self.our = our_player_index
        self.num_players = num_players
        self.hands: list[list[int]] = [[] for _ in range(num_players)]  # order ids, oldest first
        self.cards: dict[int, Card | None] = {}     # order -> Card (None if unknown to us)
        self.know: dict[int, list] = {}             # order -> [colors set, ranks set, clued]
        self.play_stacks = {c: 0 for c in SUITS}
        self.discard_pile: list[Card] = []
        self.clue_tokens = 8
        self.strikes = 0
        self.current_player_index = -1
        self.turn = 0
        self.num_drawn = 0
        self.game_over = False
        self.log: list[ActionRecord] = []
        # The play/discard whose replacement draw hasn't arrived yet (so we can
        # backfill drew_order on it -- the reactor's log-replay needs it).
        self._pending_draw: ActionRecord | None = None

    # --- incoming actions -------------------------------------------------
    def on_draw(self, player: int, order: int, suit_index: int, rank: int) -> None:
        if (self._pending_draw is not None
                and self._pending_draw.player == player
                and self._pending_draw.drew_order is None):
            self._pending_draw.drew_order = order  # the refill for the prior play/discard
            self._pending_draw = None
        self.hands[player].append(order)
        self.cards[order] = self._card(suit_index, rank)
        self.know[order] = [set(SUITS), set(RANKS), False]
        self.num_drawn += 1

    @staticmethod
    def _card(suit_index, rank) -> Card | None:
        if suit_index is not None and suit_index >= 0 and rank is not None and rank >= 1:
            return Card(SUITS[suit_index], rank)
        return None

    def _reveal(self, order, suit_index, rank) -> Card | None:
        card = self._card(suit_index, rank)
        if card is not None:
            self.cards[order] = card
        return self.cards.get(order)

    def _remove(self, player, order) -> None:
        if order in self.hands[player]:
            self.hands[player].remove(order)

    def _slot(self, player, order) -> int:
        """Index of ``order`` in the player's hand (oldest first), or 0."""
        return self.hands[player].index(order) if order in self.hands[player] else 0

    def on_play(self, player, order, suit_index=-1, rank=-1) -> None:
        card = self._reveal(order, suit_index, rank)
        idx = self._slot(player, order)
        self._remove(player, order)
        if card is not None:
            self.play_stacks[card.color] = card.rank
        rec = ActionRecord(
            turn=self.turn, player=player, action=Action.play(idx),
            played_card=card, success=True, acted_order=order)
        self.log.append(rec)
        self._pending_draw = rec

    def on_discard(self, player, order, suit_index=-1, rank=-1, failed=False) -> None:
        card = self._reveal(order, suit_index, rank)
        idx = self._slot(player, order)
        self._remove(player, order)
        if card is not None:
            self.discard_pile.append(card)
        if failed:  # a misplay -> a struck card; model it as a failed PLAY
            self.strikes += 1
            rec = ActionRecord(
                turn=self.turn, player=player, action=Action.play(idx),
                played_card=card, success=False, acted_order=order)
        else:
            rec = ActionRecord(
                turn=self.turn, player=player, action=Action.discard(idx),
                discarded_card=card, acted_order=order)
        self.log.append(rec)
        self._pending_draw = rec

    def on_clue(self, giver, target, clue_type, clue_value, touched) -> None:
        touched = set(touched)
        if clue_type == 0:  # color clue; value = suit index
            color = SUITS[clue_value]
            for order in self.hands[target]:
                kc = self.know[order]
                if order in touched:
                    kc[0], kc[2] = {color}, True
                else:
                    kc[0].discard(color)
            action = Action.clue_color(target, color)
        else:  # rank clue; value = rank
            for order in self.hands[target]:
                kc = self.know[order]
                if order in touched:
                    kc[1], kc[2] = {clue_value}, True
                else:
                    kc[1].discard(clue_value)
            action = Action.clue_rank(target, clue_value)
        self.log.append(ActionRecord(
            turn=self.turn, player=giver, action=action,
            touched_orders=tuple(touched)))

    def on_status(self, clues) -> None:
        if clues is not None:
            self.clue_tokens = clues  # authoritative count from the server

    def on_strike(self, num) -> None:
        if num is not None:
            self.strikes = num

    def on_turn(self, turn, current_player_index) -> None:
        self.turn = turn
        self.current_player_index = current_player_index

    # --- outputs ----------------------------------------------------------
    def observation(self) -> Observation:
        hands_view = []
        for p in range(self.num_players):
            cards = []
            for order in self.hands[p]:
                colors, ranks, clued = self.know[order]
                cards.append(CardView(
                    order=order,
                    card=None if p == self.our else self.cards.get(order),
                    possible_colors=frozenset(colors),
                    possible_ranks=frozenset(ranks),
                    clued=clued,
                ))
            hands_view.append(tuple(cards))
        return Observation(
            num_players=self.num_players,
            player_index=self.our,
            current_player=self.current_player_index,
            hands=tuple(hands_view),
            play_stacks=dict(self.play_stacks),
            discard_pile=tuple(self.discard_pile),
            clue_tokens=self.clue_tokens,
            max_clue_tokens=8,
            strikes=self.strikes,
            max_strikes=3,
            deck_size=max(0, DECK_SIZE - self.num_drawn),
            colors=SUITS,
            score=sum(self.play_stacks.values()),
            log=tuple(self.log),
        )

    def to_server_action(self, action: Action, table_id: int) -> dict:
        """Translate our Action into a hanab.live 'action' message body."""
        if action.type is ActionType.PLAY:
            order = self.hands[self.our][action.card_index]
            return {"tableID": table_id, "type": 0, "target": order}
        if action.type is ActionType.DISCARD:
            order = self.hands[self.our][action.card_index]
            return {"tableID": table_id, "type": 1, "target": order}
        if action.type is ActionType.CLUE_COLOR:
            return {"tableID": table_id, "type": 2, "target": action.target,
                    "value": SUITS.index(action.color)}
        # CLUE_RANK
        return {"tableID": table_id, "type": 3, "target": action.target,
                "value": action.rank}
