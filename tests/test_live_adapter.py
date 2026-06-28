"""Tests for the hanab.live adapter (no network: just the message->Observation
translation and action round-trip). The websocket client isn't imported here, so
these run without requests/websocket-client installed."""

from hanabi_sim.actions import ActionType
from hanabi_sim.cards import Color
from hanabi_sim.live.state import SUITS, LiveGameState
from hanabi_sim.players import CriticalSavePlayer


def _deal_3p(s: LiveGameState):
    """Round-robin deal of 5 cards each; our cards (player 0) are hidden (-1,-1),
    teammates' cards are visible."""
    order = 0
    for _ in range(5):
        for p in range(3):
            if p == s.our:
                s.on_draw(p, order, -1, -1)
            else:
                s.on_draw(p, order, order % 5, (order % 5) + 1)  # some valid card
            order += 1


def test_observation_built_from_messages_and_strategy_acts():
    s = LiveGameState(our_player_index=0, num_players=3)
    _deal_3p(s)
    s.on_turn(0, 0)  # our turn

    obs = s.observation()
    assert obs.num_players == 3 and obs.player_index == 0
    assert len(obs.own_hand) == 5
    assert all(cv.card is None for cv in obs.own_hand)        # our hand is hidden
    assert all(cv.card is not None for cv in obs.hands[1])    # teammates visible
    assert obs.colors == SUITS

    bot = CriticalSavePlayer()
    bot.reset()
    action = bot.act(obs)
    body = s.to_server_action(action, table_id=42)
    assert body["tableID"] == 42
    assert body["type"] in (0, 1, 2, 3)
    if body["type"] in (0, 1):              # play/discard target a card order
        assert body["target"] in s.hands[0]
    else:                                    # clues target a player + value
        assert 0 <= body["target"] < 3 and "value" in body


def test_clue_updates_card_knowledge():
    s = LiveGameState(our_player_index=0, num_players=3)
    _deal_3p(s)
    my_orders = list(s.hands[0])
    # Rank-3 clue to us touching our first card only.
    s.on_clue(giver=1, target=0, clue_type=1, clue_value=3, touched=[my_orders[0]])
    assert s.know[my_orders[0]][1] == {3}            # touched -> known rank 3
    assert 3 not in s.know[my_orders[1]][1]          # untouched -> rank 3 ruled out
    # The clue is recorded in the log for the strategies to replay.
    last = s.log[-1]
    assert last.action.type is ActionType.CLUE_RANK and last.touched_orders == (my_orders[0],)


def test_play_and_discard_update_stacks_and_pile():
    s = LiveGameState(our_player_index=0, num_players=3)
    _deal_3p(s)
    # Teammate plays a Red 1 (suit 0, rank 1).
    s.on_play(1, s.hands[1][0], suit_index=0, rank=1)
    assert s.play_stacks[Color.RED] == 1
    assert s.log[-1].success is True and s.log[-1].played_card.color is Color.RED
    # Teammate discards a card -> goes to the discard pile.
    before = len(s.discard_pile)
    s.on_discard(2, s.hands[2][0], suit_index=2, rank=4, failed=False)
    assert len(s.discard_pile) == before + 1
