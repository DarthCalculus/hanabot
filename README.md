# hanabi-sim

A fast, self-contained simulator for the cooperative card game **Hanabi**, built
to prototype and **measure the win/score rate of different strategies**.

It is *not* a client for [hanabi.live](https://github.com/Hanabi-Live/hanabi-live);
it reimplements the rules locally so thousands of games can be run per second
with no network. The hanabi.live project is the reference for the rules, and the
[hanabi-live-bot](https://github.com/Hanabi-Live/hanabi-live-bot) is the
reference for the game-state data model.

## Rules implemented

- 2–5 players; 5 suits × ranks `[1×3, 2×2, 3×2, 4×2, 5×1]` = 50 cards.
- Hand size 5 (2–3 players) or 4 (4–5 players).
- 8 clue tokens, 3 strikes.
- Actions: **play**, **discard** (only when below max tokens; regains a token),
  or **clue** a color/rank to a teammate (must touch ≥1 card; costs a token).
- Misplay → strike + card to discard. Completing a stack (playing a 5) regains a
  token (capped at 8).
- Game ends on the 3rd strike, on a perfect 25, or one full lap after the deck
  empties. By default a strikeout scores 0 (configurable).

## Layout

| Module | Purpose |
| --- | --- |
| `hanabi_sim/cards.py` | `Color`, `Card`, deck construction |
| `hanabi_sim/actions.py` | `Action` types + `ActionRecord` game log |
| `hanabi_sim/game.py` | `GameState` — the rules engine |
| `hanabi_sim/observation.py` | per-player view (own hand hidden) + inference helpers |
| `hanabi_sim/players/` | strategies (`RandomPlayer`, `GreedyPlayer`) |
| `hanabi_sim/runner.py` | run games, aggregate win/score stats |
| `hanabi_sim/cli.py` | command-line entry point |

## Usage

```bash
pip install -r requirements.txt

# Run a benchmark (self-play: every seat uses the same strategy)
python -m hanabi_sim --strategy greedy --players 3 --games 1000 --hist
python -m hanabi_sim --strategy random --players 2 --games 500

pytest        # run the test suite
```

## Writing a strategy

Subclass `Player` and implement `act(observation) -> Action`. The observation
exposes everyone else's cards, your own *clue knowledge* (not your actual
cards), the stacks, discards, tokens, strikes and the action log, plus helpers
like `is_playable`, `known_playable`, and `known_dead`.

```python
from hanabi_sim.players.base import Player
from hanabi_sim.actions import Action

class MyStrategy(Player):
    name = "mine"
    def act(self, obs):
        for i in range(len(obs.own_hand)):
            if obs.known_playable(i):
                return Action.play(i)
        ...
```

Register it in `hanabi_sim/runner.py::STRATEGIES` to use it from the CLI.

## Playing on hanab.live

`hanabi_sim/live/` is a client that plays on the real
[hanab.live](https://hanab.live) server using one of these strategies. It tracks
the game from the server's message stream, builds the same `Observation` the
strategies use, and translates their `Action` back to a server move — so the
strategy code is reused unchanged.

```bash
pip install -r requirements-live.txt          # requests + websocket-client
export HANABI_USERNAME=... HANABI_PASSWORD=... # a hanab.live bot account
python run_live.py --strategy critsave --verbose
```

Then on hanab.live: create a table and privately message the bot account
`/join`. It joins; start the game and it plays its turns. **Only the "No Variant"
(standard 5-suit) game is supported.**

Caveat: the live websocket protocol can't be exercised offline, so the message
parsing in `live/client.py` is best-effort against the official reference bot;
run with `--verbose` to see raw traffic and adjust `process_action` if a message
shape differs. The adapter itself (`live/state.py`) is unit-tested.
