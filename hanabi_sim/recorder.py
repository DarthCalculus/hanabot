"""Record a full game as a sequence of god-view snapshots for replay.

Each frame is the complete state *after* an action (frame 0 is the initial
deal), including every player's true cards, their clue-derived knowledge, the
stacks, discards, tokens and strikes, plus a description of the action that
produced the frame. The output is plain JSON-serializable dicts so it can be
embedded directly into the HTML viewer.
"""

from __future__ import annotations

from typing import Sequence, Union

from .actions import ActionRecord, ActionType
from .game import GameConfig, GameState
from .players.base import Player
from .runner import PlayerContext, PlayerFactory


def _card_str(card) -> str:
    return f"{card.color.value}{card.rank}"


def _sorted_colors(colorset, all_colors) -> list[str]:
    order = {c: i for i, c in enumerate(all_colors)}
    return [c.value for c in sorted(colorset, key=lambda c: order.get(c, 99))]


def _action_dict(rec: ActionRecord) -> dict:
    a = rec.action
    n = len(rec.touched_orders)
    plural = "s" if n != 1 else ""
    d: dict = {"player": rec.player, "type": a.type.value, "touched": list(rec.touched_orders)}
    if a.type is ActionType.PLAY:
        d["card"] = _card_str(rec.played_card)
        d["success"] = rec.success
        verb = "plays" if rec.success else "MISPLAYS"
        tag = "OK" if rec.success else "STRIKE!"
        d["text"] = f"P{rec.player} {verb} {d['card']} — {tag}"
    elif a.type is ActionType.DISCARD:
        d["card"] = _card_str(rec.discarded_card)
        d["text"] = f"P{rec.player} discards {d['card']}"
    elif a.type is ActionType.CLUE_COLOR:
        d["target"] = a.target
        d["color"] = a.color.value
        d["text"] = f"P{rec.player} clues {a.color.value} to P{a.target} ({n} card{plural})"
    else:  # CLUE_RANK
        d["target"] = a.target
        d["rank"] = a.rank
        d["text"] = f"P{rec.player} clues rank {a.rank} to P{a.target} ({n} card{plural})"
    return d


def _frame(game: GameState, rec: ActionRecord | None) -> dict:
    hands = []
    for p in range(game.num_players):
        cards = []
        for hc in game.hands[p]:
            cards.append(
                {
                    "order": hc.order,
                    "card": _card_str(hc.card),
                    "pc": _sorted_colors(hc.possible_colors, game.colors),
                    "pr": sorted(hc.possible_ranks),
                    "clued": hc.clued,
                }
            )
        hands.append(cards)
    return {
        "turn": game.turn_count,
        "current_player": game.current_player,
        "clue_tokens": game.clue_tokens,
        "strikes": game.strikes,
        "deck_size": game.deck_size,
        "score": game.score,
        "stacks": {c.value: game.play_stacks[c] for c in game.colors},
        "discard": [_card_str(c) for c in game.discard_pile],
        "hands": hands,
        "action": _action_dict(rec) if rec is not None else None,
        "game_over": game.game_over,
    }


def record_game(
    factory: Union[PlayerFactory, Sequence[PlayerFactory]],
    config: GameConfig,
    seed: int = 0,
    strategy_name: str = "custom",
) -> dict:
    """Play one game and return a replay document (JSON-serializable)."""
    if isinstance(factory, (list, tuple)):
        factories = list(factory)
        if len(factories) != config.num_players:
            raise ValueError("need exactly one factory per seat")
    else:
        factories = [factory] * config.num_players

    game = GameState(config, seed=seed)
    base = seed * 1_000
    players: list[Player] = [
        factories[i](PlayerContext(i, config.num_players, base + i, game))
        for i in range(config.num_players)
    ]
    for p in players:
        p.reset()

    frames = [_frame(game, None)]
    while not game.game_over:
        cur = game.current_player
        rec = game.apply(players[cur].act(game.observation(cur)))
        frames.append(_frame(game, rec))

    result = game.result()
    return {
        "config": {
            "num_players": config.num_players,
            "colors": [c.value for c in config.colors],
            "max_clue_tokens": config.max_clue_tokens,
            "max_strikes": config.max_strikes,
            "max_score": config.max_score,
        },
        "seed": seed,
        "strategy": strategy_name,
        "result": {
            "score": result.score,
            "stack_total": result.stack_total,
            "won": result.won,
            "strikeout": result.strikeout,
            "strikes": result.strikes,
            "turns": result.turns,
        },
        "frames": frames,
    }
