"""Run a hanab.live bot using one of our strategies.

Setup:
    pip install -r requirements-live.txt
    set HANABI_USERNAME / HANABI_PASSWORD  (env vars, or pass --username/--password)

Run:
    python run_live.py --strategy critsave --verbose

Then, on hanab.live: create a table, and from the lobby/table privately message
the bot account "/join". It joins; start the game and it plays its turns.

Only the "No Variant" (standard 5-suit) game is supported.
"""

from __future__ import annotations

import argparse
import os
import sys

from hanabi_sim.live.client import HanabiLiveClient, login
from hanabi_sim.players import (
    ChopSavePlayer,
    CriticalSavePlayer,
    FocusPlayer,
    GreedyPlayer,
    ReactorBridge4Player,
    ReactorCritPlayChopPlayer,
    ReactorDeducePlayer,
    ReactorEndgamePlayer,
    ReactorPlayer,
    ReactorPtrNoSkipPlayer,
)

STRATEGIES = {
    "rdcritplay": ReactorCritPlayChopPlayer,  # current best: chop critical/playable save
    "rdptrnoskip": ReactorPtrNoSkipPlayer,  # prior best + human-friendly pointer
    "rdbridge4": ReactorBridge4Player,
    "rdend": ReactorEndgamePlayer,
    "rdeduce": ReactorDeducePlayer,
    "reactor": ReactorPlayer,
    "critsave": CriticalSavePlayer,
    "chopsave": ChopSavePlayer,
    "focus": FocusPlayer,
    "greedy": GreedyPlayer,
}


def main() -> int:
    # hanab.live chat/table names can contain emoji; make stdout tolerate them so
    # logging never crashes the bot on a non-UTF-8 console (e.g. Windows cp1252).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Play on hanab.live with a hanabi_sim strategy")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="rdcritplay")
    parser.add_argument("--username", default=os.environ.get("HANABI_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("HANABI_PASSWORD"))
    parser.add_argument(
        "--server", default="https://hanab.live",
        help="server base URL (use e.g. http://localhost or http://localhost:8080 "
             "for a self-hosted server)",
    )
    parser.add_argument("--verbose", action="store_true", help="log raw websocket traffic")
    parser.add_argument("--reattend", type=int, default=None,
                        help="rejoin this in-progress game's table ID on connect")
    args = parser.parse_args()

    if not args.username or not args.password:
        parser.error("provide --username/--password or set HANABI_USERNAME/HANABI_PASSWORD")

    http_base = args.server.rstrip("/")
    ws_url = http_base.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    cookie = login(args.username, args.password, http_base)
    make_strategy = STRATEGIES[args.strategy]
    client = HanabiLiveClient(cookie, make_strategy, ws_url=ws_url, verbose=args.verbose,
                              reattend_table=args.reattend)
    print(f"logged in as {args.username} on {http_base}; strategy = {args.strategy}")
    client.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
