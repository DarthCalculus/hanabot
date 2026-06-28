"""Engine correctness + invariant tests."""

from __future__ import annotations

from collections import Counter

import pytest

from hanabi_sim import (
    Action,
    Card,
    Color,
    GameConfig,
    GameState,
    IllegalAction,
    build_deck,
)
from hanabi_sim.cards import RANK_COUNTS, STANDARD_COLORS
from hanabi_sim.game import HAND_SIZE
from hanabi_sim.runner import STRATEGIES, play_game, run_many


# --- deck / setup ---------------------------------------------------------
def test_deck_composition():
    deck = build_deck()
    assert len(deck) == 50
    by_color = Counter(c.color for c in deck)
    assert set(by_color) == set(STANDARD_COLORS)
    for color in STANDARD_COLORS:
        ranks = Counter(c.rank for c in deck if c.color == color)
        assert ranks == RANK_COUNTS


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_deal_hand_sizes(n):
    game = GameState(GameConfig(num_players=n), seed=0)
    for hand in game.hands:
        assert len(hand) == HAND_SIZE[n]
    # Conservation right after the deal.
    assert _total_cards(game) == 50


def _total_cards(game: GameState) -> int:
    in_hands = sum(len(h) for h in game.hands)
    played = sum(game.play_stacks.values())
    return in_hands + game.deck_size + len(game.discard_pile) + played


# --- invariants across random play ----------------------------------------
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_random_games_hold_invariants(n):
    config = GameConfig(num_players=n)
    factories = [STRATEGIES["random"]] * n
    for seed in range(40):
        game = GameState(config, seed=seed)
        players = None  # we drive via play_game for the loop instead
        # Re-run through play_game for the full loop + invariants here:
        result, finished = play_game(factories, config, seed=seed)
        assert 0 <= finished.score <= config.max_score
        assert 0 <= finished.strikes <= config.max_strikes
        assert 0 <= finished.clue_tokens <= config.max_clue_tokens
        assert finished.deck_size >= 0
        assert _total_cards(finished) == 50
        assert finished.game_over


def test_card_conservation_every_turn():
    config = GameConfig(num_players=3)
    factories = [STRATEGIES["random"]] * 3
    game = GameState(config, seed=7)
    from hanabi_sim.runner import PlayerContext

    players = [factories[i](PlayerContext(i, 3, i, game)) for i in range(3)]
    while not game.game_over:
        assert _total_cards(game) == 50
        cur = game.current_player
        game.apply(players[cur].act(game.observation(cur)))
    assert _total_cards(game) == 50


# --- determinism ----------------------------------------------------------
def test_determinism_same_seed():
    config = GameConfig(num_players=4)
    factories = [STRATEGIES["greedy"]] * 4
    r1, _ = play_game(factories, config, seed=123)
    r2, _ = play_game(factories, config, seed=123)
    assert (r1.score, r1.strikes, r1.turns) == (r2.score, r2.strikes, r2.turns)


# --- clue mechanics -------------------------------------------------------
def test_color_clue_sets_positive_and_negative_knowledge():
    game = GameState(GameConfig(num_players=2), seed=3)
    actor, target = 0, 1
    # Pick a color actually present in the target's hand.
    color = game.hands[target][0].card.color
    game.current_player = actor
    game.apply(Action.clue_color(target, color))
    for hc in game.hands[target]:
        if hc.card.color == color:
            assert hc.possible_colors == {color}
            assert hc.clued
        else:
            assert color not in hc.possible_colors


def test_rank_clue_sets_knowledge():
    game = GameState(GameConfig(num_players=2), seed=4)
    target = 1
    rank = game.hands[target][0].card.rank
    game.current_player = 0
    game.apply(Action.clue_rank(target, rank))
    for hc in game.hands[target]:
        if hc.card.rank == rank:
            assert hc.possible_ranks == {rank}
        else:
            assert rank not in hc.possible_ranks


def test_clue_must_touch_a_card():
    game = GameState(GameConfig(num_players=2), seed=5)
    target = 1
    present = {hc.card.color for hc in game.hands[target]}
    missing = next(c for c in STANDARD_COLORS if c not in present)
    with pytest.raises(IllegalAction):
        game.apply(Action.clue_color(target, missing))


def test_clue_costs_a_token_and_cluing_self_illegal():
    game = GameState(GameConfig(num_players=2), seed=6)
    rank = game.hands[1][0].card.rank
    before = game.clue_tokens
    game.apply(Action.clue_rank(1, rank))
    assert game.clue_tokens == before - 1
    game2 = GameState(GameConfig(num_players=2), seed=6)
    with pytest.raises(IllegalAction):
        game2.apply(Action.clue_rank(0, 1))  # player 0 cluing player 0


# --- play / discard / tokens ----------------------------------------------
def test_discard_illegal_at_max_tokens():
    game = GameState(GameConfig(num_players=3), seed=1)
    assert game.clue_tokens == game.config.max_clue_tokens
    with pytest.raises(IllegalAction):
        game.apply(Action.discard(0))


def test_misplay_causes_strike_and_discards_card():
    game = GameState(GameConfig(num_players=2), seed=2)
    cur = game.current_player
    # All stacks are empty, so any non-1 misplays.
    idx = next(i for i, hc in enumerate(game.hands[cur]) if hc.card.rank != 1)
    card = game.hands[cur][idx].card
    game.apply(Action.play(idx))
    assert game.strikes == 1
    assert card in game.discard_pile


def test_valid_play_advances_stack():
    game = GameState(GameConfig(num_players=2), seed=2)
    cur = game.current_player
    idx = next((i for i, hc in enumerate(game.hands[cur]) if hc.card.rank == 1), None)
    if idx is None:
        pytest.skip("no rank-1 in starting hand for this seed")
    color = game.hands[cur][idx].card.color
    game.apply(Action.play(idx))
    assert game.play_stacks[color] == 1
    assert game.strikes == 0


def test_completing_a_stack_returns_a_clue_token():
    game = GameState(GameConfig(num_players=2), seed=2)
    cur = game.current_player
    color = Color.RED
    game.play_stacks[color] = 4
    game.hands[cur][0].card = Card(color, 5)
    game.clue_tokens = 5
    game.apply(Action.play(0))
    assert game.play_stacks[color] == 5
    assert game.clue_tokens == 6  # regained on completing the stack


def test_completing_stack_at_max_tokens_does_not_overflow():
    game = GameState(GameConfig(num_players=2), seed=2)
    cur = game.current_player
    color = Color.BLUE
    game.play_stacks[color] = 4
    game.hands[cur][0].card = Card(color, 5)
    assert game.clue_tokens == game.config.max_clue_tokens
    game.apply(Action.play(0))
    assert game.clue_tokens == game.config.max_clue_tokens


# --- endgame / scoring ----------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_final_round_lasts_exactly_one_lap(n):
    """After the deck empties, the game runs exactly ``n`` more turns -- a full
    lap so the player who drew the last card also gets one final turn."""
    config = GameConfig(num_players=n)
    # Greedy never misplays, so games reliably end via deck exhaustion (the
    # scenario we want to exercise) rather than a 3rd strike.
    factories = [STRATEGIES["greedy"]] * n
    from hanabi_sim.runner import PlayerContext

    checked = 0
    for seed in range(60):
        game = GameState(config, seed=seed)
        players = [factories[i](PlayerContext(i, n, seed * 10 + i, game)) for i in range(n)]
        empty_turn = None
        while not game.game_over:
            cur = game.current_player
            game.apply(players[cur].act(game.observation(cur)))
            if game.deck_size == 0 and empty_turn is None:
                empty_turn = game.turn_count
        result = game.result()
        # Only meaningful when the game ended via deck exhaustion.
        if empty_turn is not None and not result.strikeout and result.stack_total < config.max_score:
            assert game.turn_count - empty_turn == n
            checked += 1
    assert checked > 0  # the scenario actually occurred


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_player_who_drew_last_card_gets_a_final_turn(n):
    """Regression: the drawer of the last card must act once more afterwards."""
    config = GameConfig(num_players=n)
    factories = [STRATEGIES["greedy"]] * n
    from hanabi_sim.runner import PlayerContext

    checked = 0
    for seed in range(60):
        game = GameState(config, seed=seed)
        players = [factories[i](PlayerContext(i, n, seed * 10 + i, game)) for i in range(n)]
        drawer = None
        actors_after_empty: list[int] = []
        while not game.game_over:
            cur = game.current_player
            game.apply(players[cur].act(game.observation(cur)))
            if drawer is not None:
                actors_after_empty.append(cur)
            if drawer is None and game.deck_size == 0:
                drawer = cur  # this player's action drew the last card
        result = game.result()
        if drawer is not None and not result.strikeout and result.stack_total < config.max_score:
            assert drawer in actors_after_empty
            assert len(actors_after_empty) == n  # one full lap
            checked += 1
    assert checked > 0


def test_strikeout_zeroes_score_by_default():
    game = GameState(GameConfig(num_players=2), seed=2)
    game.play_stacks[Color.RED] = 3  # some progress on the board
    game.strikes = 2
    cur = game.current_player
    # Force a misplay for the 3rd strike.
    idx = next(i for i, hc in enumerate(game.hands[cur]) if hc.card.rank != 1)
    game.apply(Action.play(idx))
    assert game.game_over and game.strikeout
    result = game.result()
    assert result.stack_total == 3
    assert result.score == 0  # standard rules zero the score


def test_strikeout_keeps_score_when_configured():
    config = GameConfig(num_players=2, loss_score_on_strikeout=False)
    game = GameState(config, seed=2)
    game.play_stacks[Color.RED] = 3
    game.strikes = 2
    cur = game.current_player
    idx = next(i for i, hc in enumerate(game.hands[cur]) if hc.card.rank != 1)
    game.apply(Action.play(idx))
    assert game.result().score == 3


# --- behavioral sanity ----------------------------------------------------
def test_greedy_beats_random_on_average():
    greedy = run_many(STRATEGIES["greedy"], num_players=3, games=120, strategy_name="greedy")
    random_ = run_many(STRATEGIES["random"], num_players=3, games=120, strategy_name="random")
    assert greedy.mean_score > random_.mean_score


def test_playclue_beats_greedy_and_does_not_strike_out_3p():
    pc = run_many(STRATEGIES["playclue"], num_players=3, games=200, strategy_name="playclue")
    gr = run_many(STRATEGIES["greedy"], num_players=3, games=200, strategy_name="greedy")
    # Generating real plays should be dramatically better than the greedy ceiling.
    assert pc.mean_score > gr.mean_score + 5
    # Singleton play clues are only ever given for genuinely playable cards.
    assert pc.strikeout_rate < 0.02


def test_ones_beats_fivesave_and_does_not_strike_out_3p():
    ones = run_many(STRATEGIES["ones"], num_players=3, games=600, strategy_name="ones")
    five = run_many(STRATEGIES["fivesave"], num_players=3, games=600, strategy_name="fivesave")
    # The multi-1 clue launches several plays at once -> a clear improvement.
    assert ones.mean_score > five.mean_score
    assert ones.strikeout_rate < 0.01


def test_deduce_five_color_by_card_counting():
    from hanabi_sim.observation import CardView, Observation
    from hanabi_sim.players.deduce_five_player import DeduceFivePlayer

    colors = STANDARD_COLORS
    my_five = CardView(order=0, card=None,
                       possible_colors=frozenset(colors), possible_ranks=frozenset({5}),
                       clued=True)
    # The opponent visibly holds four of the five 5s; only PURPLE is unaccounted for.
    others = tuple(
        CardView(order=10 + i, card=Card(c, 5),
                 possible_colors=frozenset({c}), possible_ranks=frozenset({5}), clued=False)
        for i, c in enumerate([Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE])
    )
    obs = Observation(
        num_players=2, player_index=0, current_player=0,
        hands=((my_five,), others),
        play_stacks={c: 0 for c in colors}, discard_pile=(),
        clue_tokens=8, max_clue_tokens=8, strikes=0, max_strikes=3,
        deck_size=0, colors=colors, score=0, log=(),
    )
    p = DeduceFivePlayer()
    assert p._deduced_five_color(obs, my_five) == Color.PURPLE  # only one left


def test_focus_rank4_is_stall_when_no_34_playable():
    from hanabi_sim.actions import ActionRecord
    from hanabi_sim.players.focus_player import FocusPlayer

    p = FocusPlayer()
    rec = ActionRecord(turn=0, player=0, action=Action.clue_rank(1, 4), touched_orders=(3, 7))
    early = {c: 0 for c in STANDARD_COLORS}      # nothing past 1s
    assert p._clue_play_targets(rec, early) == ()       # stall, no play target
    later = {c: 0 for c in STANDARD_COLORS}
    later[Color.RED] = 3                                # a 4 (Red) is now playable
    assert p._clue_play_targets(rec, later) == (7,)     # multi-card play -> focus


def test_focus_beats_onesdisc_3p():
    focus = run_many(STRATEGIES["focus"], num_players=3, games=500, strategy_name="focus")
    od = run_many(STRATEGIES["onesdisc"], num_players=3, games=500, strategy_name="onesdisc")
    # Multi-card play clues (focus convention) are a large jump.
    assert focus.mean_score > od.mean_score + 2
    assert focus.strikeout_rate < 0.02


def test_critsave_is_strong_and_safe_3p():
    cs = run_many(STRATEGIES["critsave"], num_players=3, games=500, strategy_name="critsave")
    assert cs.mean_score > 20.8       # critical-save convention; ~21.6 in practice
    assert cs.strikeout_rate < 0.02
    assert cs.max_score == 25         # still reaches perfect games


def test_onesdisc_does_not_strike_out_3p():
    # The count-based play/discard disambiguation for rank-1 clues must never
    # cause a misplay.
    res = run_many(STRATEGIES["onesdisc"], num_players=3, games=400, strategy_name="onesdisc")
    assert res.strikeout_rate < 0.01


def test_ones_rank1_stall_is_not_misread_as_play_regression():
    # Regression: seed 3605 once had a forced rank-1 *stall* clue read as
    # "play all your 1s", striking out on three dead 1s. The stall must avoid
    # any clue that would call a card under the active conventions.
    config = GameConfig(num_players=3)
    result, _ = play_game([STRATEGIES["ones"]] * 3, config, seed=3605)
    assert not result.strikeout


# --- replay recorder / viewer ---------------------------------------------
def test_recorder_produces_serializable_frames():
    import json

    from hanabi_sim.recorder import record_game

    config = GameConfig(num_players=3)
    replay = record_game(STRATEGIES["greedy"], config, seed=7, strategy_name="greedy")
    # frame 0 is the initial deal; one frame per action thereafter.
    assert len(replay["frames"]) == replay["result"]["turns"] + 1
    assert replay["frames"][0]["action"] is None
    assert all(f["action"] is not None for f in replay["frames"][1:])
    # Every frame conserves all 50 cards.
    for f in replay["frames"]:
        in_hands = sum(len(h) for h in f["hands"])
        played = sum(f["stacks"].values())
        assert in_hands + f["deck_size"] + len(f["discard"]) + played == 50
    # Must round-trip through JSON (it gets embedded in the HTML page).
    json.dumps(replay)


def test_viewer_renders_self_contained_html(tmp_path):
    from hanabi_sim.recorder import record_game
    from hanabi_sim.viewer import write_replay

    config = GameConfig(num_players=2)
    replay = record_game(STRATEGIES["greedy"], config, seed=1, strategy_name="greedy")
    out = write_replay(tmp_path / "r.html", replay)
    html = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "const DATA = {" in html      # data embedded, no external fetch
    assert "__DATA__" not in html        # placeholder fully substituted
