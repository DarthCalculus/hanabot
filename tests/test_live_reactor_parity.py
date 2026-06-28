"""Parity test: the live adapter must feed the reactor exactly what the local
engine does, so `rdend` plays identically on hanab.live as in self-play.

For each seed we play a local self-play game and, in lockstep, drive one
``LiveGameState`` per seat from the *broadcast* event stream (own cards hidden,
like the real server). At every turn we assert the reactor's move computed from
the live observation matches the move it made locally. If this holds, the live
bot behaves exactly like the simulated one (modulo the server actually dealing
round-robin and using these message shapes, which can't be checked offline).
"""

from __future__ import annotations

from hanabi_sim.actions import ActionType
from hanabi_sim.game import GameConfig, GameState
from hanabi_sim.live.state import SUITS, LiveGameState
from hanabi_sim.players.reactor_deduce_player import ReactorEndgamePlayer


def _suit(card) -> int:
    return SUITS.index(card.color)


def _same(a, b) -> bool:
    if a.type is not b.type:
        return False
    if a.type in (ActionType.PLAY, ActionType.DISCARD):
        return a.card_index == b.card_index
    return a.target == b.target and a.color == b.color and a.rank == b.rank


def _broadcast_draw(live, drawer, order, card):
    for s, ls in enumerate(live):
        if s == drawer:
            ls.on_draw(drawer, order, -1, -1)        # own card hidden to self
        else:
            ls.on_draw(drawer, order, _suit(card), card.rank)


def _sync(live, game):
    for ls in live:
        ls.on_status(game.clue_tokens)
        ls.on_turn(game.turn_count, game.current_player)


def test_initial_hands_reconstructs_sequential_deal():
    """hanab.live deals each player's whole hand in sequence (p0: orders 0-4,
    p1: 5-9, ...), not round-robin. The reactor must rebuild the real deal from
    the log, not assume a pattern."""
    n, H = 3, 5
    live = LiveGameState(our_player_index=0, num_players=n)
    for o in range(n * H):
        p = o // H
        if p == 0:
            live.on_draw(p, o, -1, -1)      # our own cards hidden
        else:
            live.on_draw(p, o, 0, 1)        # identities irrelevant here
    nxt = n * H
    # plays/discards at various slots/players, each with a refill draw
    live.on_discard(0, 0); live.on_draw(0, nxt, -1, -1); nxt += 1     # p0 oldest
    live.on_discard(2, 14); live.on_draw(2, nxt, 0, 1); nxt += 1      # p2 newest
    live.on_discard(1, 7); live.on_draw(1, nxt, 0, 1); nxt += 1       # p1 middle
    obs = live.observation()
    assert ReactorEndgamePlayer._initial_hands(obs) == [
        [0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]


def test_live_adapter_makes_reactor_play_identically():
    n = 3
    for seed in range(60):
        cfg = GameConfig(num_players=n)
        game = GameState(cfg, seed=seed)
        local = ReactorEndgamePlayer()
        live_players = ReactorEndgamePlayer()  # stateless; reused for all seats
        live = [LiveGameState(our_player_index=s, num_players=n) for s in range(n)]

        # Initial deal, in order-id order, hidden per seat.
        deal = sorted(
            ((p, hc.order, hc.card) for p in range(n) for hc in game.hands[p]),
            key=lambda t: t[1],
        )
        for p, order, card in deal:
            _broadcast_draw(live, p, order, card)
        _sync(live, game)

        while not game.game_over:
            cur = game.current_player
            local_action = local.act(game.observation(cur))
            live_action = live_players.act(live[cur].observation())
            assert _same(local_action, live_action), (
                f"seed {seed} turn {game.turn_count} seat {cur}: "
                f"local {local_action} vs live {live_action}")

            order = None
            if local_action.type in (ActionType.PLAY, ActionType.DISCARD):
                order = game.hands[cur][local_action.card_index].order
            rec = game.apply(local_action)

            if rec.action.type is ActionType.PLAY:
                if rec.success:
                    for ls in live:
                        ls.on_play(cur, order, _suit(rec.played_card), rec.played_card.rank)
                else:  # misplay is reported as a failed discard on hanab.live
                    for ls in live:
                        ls.on_discard(cur, order, _suit(rec.played_card),
                                      rec.played_card.rank, failed=True)
            elif rec.action.type is ActionType.DISCARD:
                for ls in live:
                    ls.on_discard(cur, order, _suit(rec.discarded_card), rec.discarded_card.rank)
            else:
                a = rec.action
                ctype = 0 if a.type is ActionType.CLUE_COLOR else 1
                cval = SUITS.index(a.color) if ctype == 0 else a.rank
                for ls in live:
                    ls.on_clue(cur, a.target, ctype, cval, rec.touched_orders)

            if rec.drew_order is not None:
                drawn = next(hc.card for hc in game.hands[cur] if hc.order == rec.drew_order)
                _broadcast_draw(live, cur, rec.drew_order, drawn)
            _sync(live, game)
