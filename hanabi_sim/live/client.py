"""A hanab.live websocket client that plays using one of our strategies.

Protocol (from the official Hanabi-Live/hanabi-live-bot reference):
  * Log in with an HTTP POST to ``/login`` (username, password, version="bot");
    the response's Set-Cookie is used to authenticate the websocket.
  * Connect to ``wss://hanab.live/ws``.
  * Messages are framed as ``"commandName {json}"`` (one space separator).
  * Get invited to a table by privately messaging the bot ``/join``; it joins,
    you start the game, and it plays its turns.

NOTE: this talks to a live server we can't exercise from here, so the exact
shapes of some game-action messages are best-effort (parsing is defensive and
``--verbose`` logs raw traffic). Adjust ``process_action`` if the live format
differs. Only the no-variant 5-suit game is supported.

Requires: ``requests`` and ``websocket-client`` (see requirements-live.txt).
"""

from __future__ import annotations

import json

import requests
import websocket

from .state import LiveGameState

PROD_HTTP = "https://hanab.live"
PROD_WS = "wss://hanab.live/ws"


def login(username: str, password: str, http_base: str = PROD_HTTP) -> str:
    """Log in and return the session cookie string for the websocket."""
    resp = requests.post(
        http_base + "/login",
        data={"username": username, "password": password, "version": "bot"},
    )
    resp.raise_for_status()
    cookie = resp.headers.get("Set-Cookie")
    if not cookie:
        raise RuntimeError("login succeeded but no Set-Cookie was returned")
    return cookie


class HanabiLiveClient:
    def __init__(self, cookie, make_strategy, ws_url=PROD_WS, verbose=False,
                 reattend_table=None):
        self.cookie = cookie
        self.make_strategy = make_strategy  # () -> Player
        self.verbose = verbose
        self.reattend_table = reattend_table  # rejoin this in-progress game on connect
        self.username = None
        self.tables: dict[int, dict] = {}
        self.users: dict[str, int] = {}  # username -> tableID they're seated at
        self.table_id = None
        self.state: LiveGameState | None = None
        self.strategy = None
        self.replaying = False
        self.last_current = -1  # last current-player seen (to detect a fresh turn)
        self.all_connected = True  # don't act until every seat has loaded the game
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            cookie=cookie,
        )

    def run(self):
        # Send a WebSocket ping every 10s. hanab.live drops a client that goes
        # silent (~15s), so an idle lobby bot would otherwise be disconnected
        # before anyone can invite it; the keepalive keeps the connection open.
        self.ws.run_forever(ping_interval=10, ping_timeout=8)

    # --- transport --------------------------------------------------------
    @staticmethod
    def _safe_print(*parts) -> None:
        """Print without ever raising on a non-encodable console (e.g. an emoji
        on a Windows cp1252 terminal) -- otherwise an exception here kills the
        websocket loop and freezes the bot."""
        text = " ".join(str(p) for p in parts)
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("ascii", "replace").decode("ascii"))

    def _send(self, command: str, data: dict):
        if self.verbose:
            self._safe_print(">>", command, data)
        self.ws.send(command + " " + json.dumps(data))

    def _on_open(self, ws):
        print("connected to hanab.live; PM the bot '/join' from your table to invite it")

    def _on_error(self, ws, error):
        self._safe_print("ws error:", error)

    def _on_close(self, ws, *args):
        print("ws closed")

    def _on_message(self, ws, message):
        if self.verbose:
            self._safe_print("<<", message[:500])
        command, _, body = message.partition(" ")
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return
        handler = getattr(self, "cmd_" + command, None)
        if handler is not None:
            # A bad message must never freeze the bot -- log and keep going.
            try:
                handler(data)
            except Exception as e:  # pragma: no cover - defensive
                self._safe_print(f"error handling {command}: {e!r}")

    # --- lobby commands ---------------------------------------------------
    def cmd_welcome(self, data):
        self.username = data.get("username")
        if self.reattend_table is not None:
            # Try to rejoin an in-progress game we were a player in.
            print(f"attempting to reattend table {self.reattend_table}")
            self._send("tableReattend", {"tableID": self.reattend_table})

    def cmd_table(self, data):
        self.tables[data["id"]] = data

    def cmd_tableList(self, data):
        for t in data if isinstance(data, list) else []:
            self.tables[t["id"]] = t

    def cmd_tableGone(self, data):
        self.tables.pop(data.get("id"), None)

    def cmd_user(self, data):
        if data.get("name"):
            self.users[data["name"]] = data.get("tableID", 0)

    def cmd_userList(self, data):
        for u in data if isinstance(data, list) else []:
            if u.get("name"):
                self.users[u["name"]] = u.get("tableID", 0)

    def _try_join_from_chat(self, data):
        msg = (data.get("msg") or "").strip()
        who = data.get("who") or data.get("username")
        if "/join" not in msg:
            return
        # Join the table the inviter is currently seated at.
        target = self.users.get(who, 0)
        if not target:  # fall back to the players list of a known table
            for tid, t in self.tables.items():
                if who in t.get("players", []):
                    target = tid
        if target:
            print(f"joining table {target} (invited by {who})")
            self._send("tableJoin", {"tableID": target})
        else:
            print(f"got /join from {who} but couldn't find their table; "
                  f"are you seated at a table? (known users: {self.users})")

    def cmd_chat(self, data):
        self._try_join_from_chat(data)

    def cmd_chatPM(self, data):
        self._try_join_from_chat(data)

    def cmd_warning(self, data):
        print("warning:", data.get("warning"))

    def cmd_error(self, data):
        print("error:", data.get("error"))

    # --- game setup -------------------------------------------------------
    def cmd_tableStart(self, data):
        self.table_id = data.get("tableID", self.table_id)
        self._send("getGameInfo1", {"tableID": self.table_id})

    def cmd_init(self, data):
        self.table_id = data.get("tableID", self.table_id)
        names = data.get("playerNames", [])
        our = data.get("ourPlayerIndex", 0)
        variant = (data.get("options") or {}).get("variantName", "No Variant")
        if variant != "No Variant":
            print(f"WARNING: variant is {variant!r}; this bot only handles 'No Variant'.")
        self.state = LiveGameState(our, len(names))
        self.last_current = -1
        self.strategy = self.make_strategy()
        if hasattr(self.strategy, "reset"):
            self.strategy.reset()
        self._send("getGameInfo2", {"tableID": self.table_id})

    def cmd_gameActionList(self, data):
        actions = data.get("list", data if isinstance(data, list) else [])
        self.replaying = True
        for a in actions:
            self.process_action(a)
        self.replaying = False
        self._send("loaded", {"tableID": self.table_id})
        self._maybe_act()

    def cmd_gameAction(self, data):
        self.process_action(data.get("action", data))
        self._maybe_act()

    def cmd_clock(self, data):
        # The active player is reported here; at game start there's no separate
        # "turn" action, so this is how the first turn is signalled.
        if self.state is not None and data.get("activePlayerIndex") is not None:
            self.state.current_player_index = data["activePlayerIndex"]
            self._maybe_act()

    def cmd_connected(self, data):
        # Per-seat "has loaded the game" flags. Don't act until everyone is in,
        # so a fast starting player can't move before others have loaded and miss
        # broadcasting its move to them (which desyncs the late-loaders).
        lst = data.get("list")
        self.all_connected = all(lst) if lst else True
        self._maybe_act()

    # --- core game loop ---------------------------------------------------
    def process_action(self, a: dict):
        s = self.state
        if s is None:
            return
        t = a.get("type")
        if t == "draw":
            s.on_draw(a["playerIndex"], a["order"], a.get("suitIndex", -1), a.get("rank", -1))
        elif t == "play":
            w = a.get("which", a)
            s.on_play(w.get("playerIndex", a.get("playerIndex")), w["order"],
                      w.get("suitIndex", a.get("suitIndex", -1)), w.get("rank", a.get("rank", -1)))
        elif t == "discard":
            w = a.get("which", a)
            s.on_discard(w.get("playerIndex", a.get("playerIndex")), w["order"],
                         w.get("suitIndex", a.get("suitIndex", -1)),
                         w.get("rank", a.get("rank", -1)), a.get("failed", False))
        elif t == "clue":
            c = a["clue"]
            s.on_clue(a["giver"], a["target"], c["type"], c["value"], a["list"])
        elif t == "strike":
            s.on_strike(a.get("num"))
        elif t == "status":
            s.on_status(a.get("clues"))
        elif t == "turn":
            s.on_turn(a.get("num", s.turn), a["currentPlayerIndex"])
        elif t in ("gameOver", "playerTimes"):
            s.game_over = True

    def _maybe_act(self):
        s = self.state
        # Note: the all_connected gate is checked before touching last_current, so
        # a gated call doesn't consume the "fresh turn" edge.
        if s is None or self.replaying or s.game_over or not self.all_connected:
            return
        cur = s.current_player_index
        # Act only when it *becomes* our turn -- not on the stale echoes the
        # server sends after our move but before the turn officially flips.
        fresh = cur == s.our and self.last_current != s.our
        self.last_current = cur
        if not fresh:
            return
        obs = s.observation()
        action = self.strategy.act(obs)
        body = s.to_server_action(action, self.table_id)
        print("decided:", action)
        self._send("action", body)
