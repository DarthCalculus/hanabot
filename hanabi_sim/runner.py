"""Run games and aggregate results for strategy evaluation."""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .game import GameConfig, GameResult, GameState
from .players.base import Player
from .players.chop_first_player import ChopFirstPlayer
from .players.chop_save_player import ChopSavePlayer
from .players.critical_save_player import CriticalSavePlayer
from .players.deduce_five_player import DeduceFivePlayer
from .players.distant_save_player import DistantSavePlayer
from .players.five_save_player import FiveSavePlayer
from .players.focus_player import FocusPlayer
from .players.good_touch_player import GoodTouchPlayer
from .players.greedy_player import GreedyPlayer
from .players.ones_discard_player import OnesDiscardPlayer
from .players.ones_player import OnesPlayer
from .players.play_clue_player import PlayCluePlayer
from .players.player_first_player import PlayerFirstPlayer
from .players.random_player import RandomPlayer
from .players.reactor_deduce_player import (
    ReactorDeducePlayer,
    ReactorEndgamePlayer,
    ReactorScoredPlayer,
    ReactorBridgePlayer,
    ReactorBridge4Player,
    ReactorPtrNoSkipPlayer,
    ReactorCritPlayChopPlayer,
    ReactorSafeCmdPlayer,
    ReactorByInitPlayer,
    ReactorCond2ViewPlayer,
    ReactorReactDiscDedupPlayer,
    ReactorNoChopDiscardPlayer,
    ReactorLockPlayer,
)
from .players.reactor_player import ReactorPlayer
from .players.tempo_player import TempoPlayer

# A factory builds the Player for one seat. It receives a read-only context so
# strategies that need legal actions or a deterministic seed can get them
# without coupling to GameState internals.


@dataclass
class PlayerContext:
    index: int
    num_players: int
    seed: int
    game: GameState  # read-only access (e.g. legal_actions) — do not mutate

    def legal_actions(self, obs):
        return self.game.legal_actions(obs.player_index)


PlayerFactory = Callable[[PlayerContext], Player]

# Names usable on the CLI / in benchmarks.
STRATEGIES: dict[str, PlayerFactory] = {
    "random": lambda ctx: RandomPlayer(ctx.legal_actions, seed=ctx.seed),
    "greedy": lambda ctx: GreedyPlayer(),
    "playclue": lambda ctx: PlayCluePlayer(),
    "fivesave": lambda ctx: FiveSavePlayer(),
    "chopfirst": lambda ctx: ChopFirstPlayer(),
    "ones": lambda ctx: OnesPlayer(),
    "onesdisc": lambda ctx: OnesDiscardPlayer(),
    "focus": lambda ctx: FocusPlayer(),
    "deduce5": lambda ctx: DeduceFivePlayer(),
    "goodtouch": lambda ctx: GoodTouchPlayer(),
    "tempo": lambda ctx: TempoPlayer(),
    "critsave": lambda ctx: CriticalSavePlayer(),
    "chopsave": lambda ctx: ChopSavePlayer(),
    "pfirst": lambda ctx: PlayerFirstPlayer(),
    "distsave": lambda ctx: DistantSavePlayer(),
    "reactor": lambda ctx: ReactorPlayer(),
    "rdeduce": lambda ctx: ReactorDeducePlayer(),
    "rdend": lambda ctx: ReactorEndgamePlayer(),
    "rdscore": lambda ctx: ReactorScoredPlayer(),
    "rdbridge": lambda ctx: ReactorBridgePlayer(),
    "rdbridge4": lambda ctx: ReactorBridge4Player(),
    "rdptrnoskip": lambda ctx: ReactorPtrNoSkipPlayer(),
    "rdcritplay": lambda ctx: ReactorCritPlayChopPlayer(),
    "rdcmd": lambda ctx: ReactorSafeCmdPlayer(),
    "rdinit": lambda ctx: ReactorByInitPlayer(),
    "rdc2v": lambda ctx: ReactorCond2ViewPlayer(),
    "rdrdd": lambda ctx: ReactorReactDiscDedupPlayer(),
    "rdncd": lambda ctx: ReactorNoChopDiscardPlayer(),
    "rdlock": lambda ctx: ReactorLockPlayer(),
}

# Guard against a strategy that never makes progress.
MAX_TURNS = 5000


def play_game(
    factories: Sequence[PlayerFactory],
    config: GameConfig,
    seed: int | None = None,
) -> tuple[GameResult, GameState]:
    """Play one full game. ``factories`` has one entry per seat."""
    if len(factories) != config.num_players:
        raise ValueError("need exactly one factory per seat")

    game = GameState(config, seed=seed)
    base = 0 if seed is None else seed * 1_000
    players = [
        factories[i](PlayerContext(i, config.num_players, base + i, game))
        for i in range(config.num_players)
    ]
    for p in players:
        p.reset()

    turns = 0
    while not game.game_over:
        cur = game.current_player
        action = players[cur].act(game.observation(cur))
        game.apply(action)
        turns += 1
        if turns > MAX_TURNS:  # pragma: no cover - safety net
            raise RuntimeError("game exceeded MAX_TURNS; strategy made no progress")
    return game.result(), game


def _play_chunk(args):
    """Worker: play a chunk of seeds with a registered strategy (self-play).
    Takes/returns only picklable data so it works across processes."""
    strategy_name, num_players, seeds = args
    config = GameConfig(num_players=num_players)
    factories = [STRATEGIES[strategy_name]] * num_players
    stacks: list[int] = []
    won = strike = turns = 0
    for seed in seeds:
        result, _ = play_game(factories, config, seed=seed)
        stacks.append(result.stack_total)
        won += int(result.won)
        strike += int(result.strikeout)
        turns += result.turns
    return stacks, won, strike, turns


def run_many_parallel(
    strategy_name: str,
    num_players: int,
    games: int,
    base_seed: int = 1,
    jobs: int | None = None,
) -> "Summary":
    """Same self-play benchmark as ``run_many`` but spread across processes.

    Deterministic and identical in aggregate to ``run_many`` for the same
    (strategy, base_seed, games). ``strategy_name`` must be in ``STRATEGIES``
    (workers rebuild it by name -- lambdas/local classes can't cross processes).
    """
    import concurrent.futures
    import os

    jobs = jobs or max(1, (os.cpu_count() or 2) - 2)
    seeds = list(range(base_seed, base_seed + games))
    chunks = [seeds[i::jobs] for i in range(jobs)]
    args = [(strategy_name, num_players, ch) for ch in chunks if ch]

    all_stacks: list[int] = []
    won = strike = turns = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
        for stacks, w, s, t in ex.map(_play_chunk, args):
            all_stacks.extend(stacks)
            won += w
            strike += s
            turns += t

    return Summary(
        strategy=strategy_name,
        num_players=num_players,
        games=games,
        mean_score=statistics.fmean(all_stacks),
        stdev_score=statistics.pstdev(all_stacks) if len(all_stacks) > 1 else 0.0,
        min_score=min(all_stacks),
        max_score=max(all_stacks),
        perfect_rate=won / games,
        strikeout_rate=strike / games,
        mean_turns=turns / games,
        score_hist=Counter(all_stacks),
    )


@dataclass
class Summary:
    strategy: str
    num_players: int
    games: int
    mean_score: float
    stdev_score: float
    min_score: int
    max_score: int
    perfect_rate: float       # fraction scoring the maximum
    strikeout_rate: float     # fraction ending on the 3rd strike
    mean_turns: float
    score_hist: Counter

    def format(self) -> str:
        lines = [
            f"Strategy: {self.strategy}   Players: {self.num_players}   Games: {self.games}",
            f"  Mean score : {self.mean_score:.3f}  (sd {self.stdev_score:.3f})",
            f"  Score range: {self.min_score}..{self.max_score}",
            f"  Perfect    : {self.perfect_rate * 100:.2f}%",
            f"  Strikeouts : {self.strikeout_rate * 100:.2f}%",
            f"  Mean turns : {self.mean_turns:.1f}",
        ]
        return "\n".join(lines)

    def histogram(self, width: int = 40) -> str:
        if not self.score_hist:
            return ""
        peak = max(self.score_hist.values())
        rows = []
        for s in range(min(self.score_hist), max(self.score_hist) + 1):
            n = self.score_hist.get(s, 0)
            bar = "#" * round(width * n / peak) if peak else ""
            rows.append(f"  {s:2d} | {bar} {n}")
        return "\n".join(rows)


def run_many(
    factory: PlayerFactory,
    num_players: int,
    games: int,
    config: GameConfig | None = None,
    base_seed: int = 1,
    strategy_name: str = "custom",
) -> Summary:
    """Run ``games`` self-play games (every seat uses the same factory)."""
    if config is None:
        config = GameConfig(num_players=num_players)
    factories = [factory] * num_players

    results: list[GameResult] = []
    for g in range(games):
        result, _ = play_game(factories, config, seed=base_seed + g)
        results.append(result)

    # Report against the realized stack total so the strikeout penalty doesn't
    # hide a strategy's progress; strikeouts are tracked separately.
    scores = [r.stack_total for r in results]
    hist = Counter(scores)
    return Summary(
        strategy=strategy_name,
        num_players=num_players,
        games=games,
        mean_score=statistics.fmean(scores),
        stdev_score=statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        min_score=min(scores),
        max_score=max(scores),
        perfect_rate=sum(1 for r in results if r.won) / games,
        strikeout_rate=sum(1 for r in results if r.strikeout) / games,
        mean_turns=statistics.fmean(r.turns for r in results),
        score_hist=hist,
    )
