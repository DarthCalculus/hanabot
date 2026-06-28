"""Tests for the Reactor bot's signal mechanics."""

from __future__ import annotations

from hanabi_sim.cards import Color
from hanabi_sim.game import GameConfig, GameState
from hanabi_sim.players.reactor_deduce_player import ReactorDeducePlayer
from hanabi_sim.players.reactor_player import ReactorPlayer
from hanabi_sim.runner import STRATEGIES, play_game


def _slots(orders):
    return ReactorPlayer._hand_slots(orders)


# Newest card (highest order) is slot 1; oldest is the highest slot.
def test_hand_slots_newest_is_slot_one():
    slots, by = _slots([7, 13, 10])
    assert slots == {13: 1, 10: 2, 7: 3}
    assert by == {1: 13, 2: 10, 3: 7}


def test_point_left_basic_and_wraparound():
    R = ReactorPlayer()
    _slots_, by = _slots([0, 1, 2, 3, 4])  # 5 slots, all untouched
    # one step left from slot 3 -> slot 2
    assert R._point_left(3, by, 5, set()) == 2
    # from the leftmost slot 1, wrap around to the oldest slot 5
    assert R._point_left(1, by, 5, set()) == 5


def test_point_left_skips_touched():
    R = ReactorPlayer()
    _slots_, by = _slots([0, 1, 2, 3, 4])
    touched = {by[2], by[3]}  # slots 2 and 3 are clued
    # from slot 4, skip 3 and 2, land on slot 1
    assert R._point_left(4, by, 5, touched) == 1


def test_point_left_all_touched_returns_self():
    R = ReactorPlayer()
    _slots_, by = _slots([0, 1])
    touched = {0, 1}
    assert R._point_left(1, by, 2, touched) == 1


# orders [0..4] -> slots 4:1, 3:2, 2:3, 1:4, 0:5
def test_initial_color_new_plays_leftmost_new():
    R = ReactorPlayer()
    orders = [0, 1, 2, 3, 4]
    # touch orders 3 (slot 2) and 1 (slot 4); leftmost new = slot 2
    assert R._compute_initial(orders, [3, 1], set(), {}, {}, True) == ("play", 2)


def test_initial_rank_single_new_points_left():
    R = ReactorPlayer()
    orders = [0, 1, 2, 3, 4]
    # touch order 3 (slot 2); pointer = first untouched to the left = slot 1
    assert R._compute_initial(orders, [3], set(), {}, {}, False) == ("discard", 1)


def test_initial_rank_multi_new_leftmost_pointer():
    R = ReactorPlayer()
    orders = [0, 1, 2, 3, 4]
    # touch 3 (slot2)->points slot1 ; 1 (slot4)->points slot3 ; leftmost = slot1
    assert R._compute_initial(orders, [3, 1], set(), {}, {}, False) == ("discard", 1)


def test_initial_no_new_touch_branches():
    R = ReactorPlayer()
    orders = [0, 1, 2, 3, 4]
    o = 3  # slot 2
    # color clue, color not yet known -> new info -> play
    assert R._compute_initial(orders, [o], {o}, {}, {o: 2}, True) == ("play", 2)
    # color clue, color already known, rank unknown -> discard
    assert R._compute_initial(orders, [o], {o}, {o: Color.RED}, {}, True) == ("discard", 2)
    # both known: color clue -> play, rank clue -> discard
    assert R._compute_initial(orders, [o], {o}, {o: Color.RED}, {o: 2}, True) == ("play", 2)
    assert R._compute_initial(orders, [o], {o}, {o: Color.RED}, {o: 2}, False) == ("discard", 2)
    # rank clue, rank not yet known -> new info -> play
    assert R._compute_initial(orders, [o], {o}, {o: Color.RED}, {}, False) == ("play", 2)


def test_reactor_self_play_3p_runs_clean():
    """A batch of 3-player self-play games complete and stay in range."""
    config = GameConfig(num_players=3)
    factory = STRATEGIES["reactor"]
    for seed in range(40):
        result, _ = play_game([factory] * 3, config, seed=seed)
        assert 0 <= result.stack_total <= 25


def test_deduced_plays_are_always_truly_playable():
    """The card-counting deduction must be sound: any card it elects to play is
    genuinely playable (checked against god-view), so it can never misplay."""
    cfg = GameConfig(num_players=3)
    checked = 0
    for seed in range(80):
        game = GameState(cfg, seed=seed)
        players = [ReactorDeducePlayer() for _ in range(3)]
        for p in players:
            p.reset()
        while not game.game_over:
            cur = game.current_player
            obs = game.observation(cur)
            pl = players[cur]
            extra = pl._extra_play(obs, pl._derive(obs))
            if extra is not None and extra.card_index is not None:
                card = game.hands[cur][extra.card_index].card
                assert card.rank == game.play_stacks[card.color] + 1, (
                    f"seed {seed}: deduced an unplayable {card}")
                checked += 1
            game.apply(pl.act(obs))
    assert checked > 0  # the deduction actually fired somewhere


def test_trash_discards_are_truly_dead():
    """Any card chosen by the provable-trash discard is genuinely dead."""
    cfg = GameConfig(num_players=3)
    checked = 0
    for seed in range(80):
        game = GameState(cfg, seed=seed)
        players = [ReactorDeducePlayer() for _ in range(3)]
        for p in players:
            p.reset()
        while not game.game_over:
            cur = game.current_player
            obs = game.observation(cur)
            pl = players[cur]
            td = pl._trash_discard(obs)
            if td is not None:
                card = game.hands[cur][td.card_index].card
                assert card.rank <= game.play_stacks[card.color], (
                    f"seed {seed}: trash-discarded a live {card}")
                checked += 1
            game.apply(pl.act(obs))
    assert checked > 0


def test_reactor_other_counts_do_not_crash():
    for n in (2, 4, 5):
        config = GameConfig(num_players=n)
        factory = STRATEGIES["reactor"]
        result, _ = play_game([factory] * n, config, seed=1)
        assert 0 <= result.stack_total <= 25
