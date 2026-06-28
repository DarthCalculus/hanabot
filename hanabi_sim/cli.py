"""Command-line entry point.

Benchmark a strategy:
    python -m hanabi_sim --strategy greedy --players 3 --games 1000 --hist

Generate a step-through HTML replay of a single game:
    python -m hanabi_sim --replay game.html --strategy greedy --players 3 --seed 7 --open
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from .game import GameConfig
from .recorder import record_game
from .runner import STRATEGIES, run_many, run_many_parallel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hanabi strategy simulator")
    parser.add_argument(
        "--strategy", choices=sorted(STRATEGIES), default="greedy",
        help="strategy used by every seat (self-play)",
    )
    parser.add_argument("--players", type=int, default=3, choices=[2, 3, 4, 5])
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1, help="base seed")
    parser.add_argument("--hist", action="store_true", help="print a score histogram")
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel worker processes for benchmarking "
                             "(1 = serial, 0 = auto = cores-2)")
    parser.add_argument(
        "--replay", nargs="?", const="replay.html", default=None, metavar="FILE",
        help="record ONE game (using --strategy/--players/--seed) to an HTML "
             "step-through viewer instead of benchmarking (default: replay.html)",
    )
    parser.add_argument("--open", action="store_true", help="open the replay in a browser")
    args = parser.parse_args(argv)

    config = GameConfig(num_players=args.players)

    if args.replay is not None:
        from .viewer import write_replay

        replay = record_game(
            STRATEGIES[args.strategy], config, seed=args.seed,
            strategy_name=args.strategy,
        )
        out = write_replay(args.replay, replay)
        r = replay["result"]
        outcome = "strikeout (0)" if r["strikeout"] else (
            "perfect 25" if r["won"] else f"score {r['stack_total']}")
        print(f"Wrote replay -> {out.resolve()}  ({len(replay['frames']) - 1} actions, {outcome})")
        if args.open:
            webbrowser.open(out.resolve().as_uri())
        return 0

    if args.jobs != 1:
        summary = run_many_parallel(
            args.strategy,
            num_players=args.players,
            games=args.games,
            base_seed=args.seed,
            jobs=(None if args.jobs == 0 else args.jobs),
        )
    else:
        summary = run_many(
            STRATEGIES[args.strategy],
            num_players=args.players,
            games=args.games,
            config=config,
            base_seed=args.seed,
            strategy_name=args.strategy,
        )
    print(summary.format())
    if args.hist:
        print("\nScore distribution:")
        print(summary.histogram())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
