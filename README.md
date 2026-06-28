# hanabot

A fast, self-contained simulator for the cooperative card game **Hanabi**, a
ladder of convention-based bot strategies (culminating in the **reactor** line),
an HTML replay viewer, and a client that lets the bots play on
[hanab.live](https://hanab.live).

The best strategy, **`rdend`**, wins a perfect game ~**76%** of the time in
3-player self-play (mean score ~24.6/25, and it never strikes out).

---

## Contents

- [Install](#install)
- [Running simulations](#running-simulations)
- [Playing with the bot on hanab.live](#playing-with-the-bot-on-hanablive)
- [How the reactor bot works — a player's guide](#how-the-reactor-bot-works--a-players-guide)
- [Project layout](#project-layout)

---

## Install

```bash
pip install -r requirements.txt          # core simulator (pure standard library)
pip install -r requirements-live.txt     # only if you want to play on hanab.live
```

Python 3.10+ recommended.

---

## Running simulations

Everything runs through `python -m hanabi_sim`.

**Benchmark a strategy** (every seat plays the same strategy):

```bash
python -m hanabi_sim --strategy rdend --players 3 --games 20000 --jobs 8
```

- `--strategy` — which bot to test (`rdend` is the strongest; see the list below).
- `--players` — 2–5. (The reactor strategies are built for **3 players**.)
- `--games` — how many games to play.
- `--jobs` — spread the games across N worker processes for speed (`0` = auto).
- `--hist` — also print a score histogram.

It reports the mean score, perfect-game (win) rate, strikeout rate, and more.

**Watch a single game in your browser** — a step-through replay viewer:

```bash
python -m hanabi_sim --replay game.html --strategy rdend --players 3 --seed 7 --open
```

This writes a self-contained `game.html` and opens it. Step through every move with
the arrow keys (or the buttons); each player's hand, what they know, the stacks,
and the discard pile are shown at each step. Cards are drawn **newest-on-the-left**
to match the bot's convention (explained below).

**Run the tests:**

```bash
pytest
```

### Strategies available

The interesting ones, roughly weakest → strongest:

| name | idea |
| --- | --- |
| `greedy` | only plays cards it's 100% sure of; no conventions |
| `playclue` | a single-card clue means "play it" |
| `focus` / `deduce5` | multi-card play clues + card-counting your own 5s |
| `critsave` | the above, plus saving the last copy of a still-needed card |
| `reactor` | the reactor convention (explained below) |
| `rdeduce` | reactor + proves its own cards playable by card-counting |
| **`rdend`** | **rdeduce + endgame play-prioritisation + last-turn gambles — the best** |

(More experimental ones are registered in `hanabi_sim/runner.py`.)

---

## Playing with the bot on hanab.live

You'll need a **bot account** on [hanab.live](https://hanab.live) for each bot seat
(just register normal accounts for them).

```bash
pip install -r requirements-live.txt
python run_live.py --strategy rdend --username YOUR_BOT --password YOUR_BOT_PW --verbose
```

(Credentials can also come from the `HANABI_USERNAME` / `HANABI_PASSWORD`
environment variables instead of the command line.)

Then, on hanab.live:

1. Start the script once **per bot seat** you want to fill. The reactor wants a
   **3-player** game, so for you + 2 bots, run it twice (one per bot account).
2. Create a table. It **must be "No Variant"** (the standard 5-suit game) — the
   bots don't understand variants.
3. Sit down at the table, then privately message each bot `/pm THE_BOT_NAME /join`
   (`/w` also works). Be seated at the table **before** you send `/join` so the bot
   can find it.
4. Start the game — the bots take their own turns automatically.

If a game gets stuck (e.g. a bot disconnects), restart that bot with
`--reattend <tableID>` to rejoin the game in progress.

> **Heads-up:** the reactor bots only work in a **3-player, No-Variant** game, and
> they assume **everyone at the table plays by the convention below.** Playing with
> them successfully means learning that convention — so read on.

---

## How the reactor bot works — a player's guide

This is the convention `rdend` plays by. If you want to take the third (human) seat
and not accidentally make the bots bomb, this is the section to read. It looks like
a lot, but the day-to-day rules are short — most of the length is making the one
tricky part (reactive clues) crystal clear.

### 1. The two building blocks: slots and signals

**Slots.** You draw new cards onto the **left** of your hand, so your **newest**
card is **slot 1** and the numbers climb as cards get older:

```
 slot 1     slot 2     slot 3     slot 4     slot 5
(newest) <------------------------------> (oldest)
```

**Signals.** During the game, cards pick up **signals** that the whole team tracks
(they come from clues — see below):

- a **play signal** — "this card is playable; play it."
- a **discard signal** — "this card is safe to throw away."

On your turn you just act on your signals. The entire convention is really just **a
system for putting the right signal on the right card.**

### 2. Clues: *who* you clue decides what the clue means

This is the one idea that makes the reactor different from ordinary Hanabi. In a
3-player game, on your turn the two other players are:

- the **next** player (acts immediately after you), and
- the **far** player (acts after that — the one whose turn does *not* come next).

**Which one you clue changes how the clue is read:**

- A clue to the **next** player is a **stable clue** — it puts a signal straight
  onto their hand. Simple.
- A clue to the **far** player is a **reactive clue** — a team move that also
  involves the player in between. Powerful, but handle with care (Section 4).

### 3. Stable clues — cluing the player right after you

- **A color clue means "play."** It puts a play signal on the **newest** card the
  clue newly touches.
- **A number clue means "discard."** It puts a discard signal on the card just to
  the **left (newer side)** of the clued card — skipping any already-clued cards,
  and wrapping from slot 1 around to the oldest card. (Think of a number as
  pointing: "the card next to this one is the safe thing to throw.")

Two special cases:

- **Number 1 means "play all my 1s"** — cluing 1s says play every 1 it touches
  (unless there are obviously too many to all be playable, in which case: discard
  them).
- **In the endgame, a number 4 or 5 can mean "stall"** — see Section 6.

> Quick version: for the next player, **color = play, number = discard** (except
> 1s, and endgame 4s/5s).

### 4. Reactive clues — cluing the *far* player (the important part)

A reactive clue goes to the player **two seats ahead** of you (not the one who
plays next). It's a **two-step message**, and the player **between** you and the
target is part of it.

*Why do it this way?* It lets the in-between player's ordinary turn do double duty:
they make a useful play or discard **and** relay your message at the same time, so
the team communicates without "wasting" a turn.

**Step 1 — the clue points at a card.** Read by the **same rules as a stable clue
(Section 3)** — color = a play, number = a discard — the clue *tentatively* points
at one card in the **far** player's hand.

**Step 2 — the in-between player edits it with their move.** On their turn, that
player makes a real play or discard, and *how* they do it adjusts the tentative
instruction:

- **Play vs discard sets the kind.** If they **play** a card, the instruction keeps
  its kind (a tentative "play" stays a "play"). If they **discard**, it **flips**
  ("play" ↔ "discard").
- **Which card they act on moves the target.** If they act on their **newest** card
  (slot 1), the target doesn't move. For each slot **older** they act on instead,
  the far player's target shifts **one card newer** (wrapping around).

**Step 3 — the far player carries out the final instruction** on their turn.

#### ⚠️ The golden rule (this is what makes or breaks it)

The in-between player is **forced** to play or discard in order to relay your
message. So **only give a reactive clue when that forced move is genuinely safe for
them:**

- if the message needs them to **play**, they must have a truly **playable** card
  in the slot it lands on;
- if it needs them to **discard**, that slot must be **junk** (or a spare
  duplicate).

If you give a reactive clue that forces them to play a card that *isn't* playable,
**they will bomb.** The bots check this automatically before ever giving a reactive
clue — so when a *bot* gives one, trust that the reaction is safe. As the human,
**you** must check it yourself. When in doubt, just give a plain stable clue to
your next neighbor instead.

#### A worked example

Turn order: **You → Bob → Cathy.** You want Cathy to play her slot-1 card (say it's
a playable Red 1).

1. You give Cathy a **color** clue (color = "play"). Suppose it tentatively points
   at Cathy's **slot 2** (the newest card your clue touched).
2. The target needs to move from slot 2 to slot 1 — one card newer. Bob relays that
   "shift by one" by acting on **his** slot 2 (one older than newest), and since
   the instruction should stay a "play," Bob **plays** (rather than discards).
3. So Bob plays his slot 2 — **which only works if you first made sure Bob's slot 2
   is a card he can actually play.** Cathy then sees: a color clue (tentative "play
   slot 2") + Bob playing his slot 2 (keep the kind, shift one newer) → **play slot
   1.** She plays her Red 1. 🎉

(If Cathy instead had no playable card but some junk, you'd aim the message at a
"discard," and Bob would relay it by **discarding** the appropriate slot.)

#### What to do in each seat

- **You're giving the clue** → prefer the next player (stable and easy). Only clue
  the far player once you've checked the in-between player can react safely.
- **You're in the middle** (someone clued the player after you) → you **must react
  this turn**, and your play/discard *is* the relay. This is the fiddliest seat, so
  here's the exact recipe (bots do it automatically; it's only on you when you're
  the human there):
    1. Read the clue's **tentative** instruction in the far player's hand (kind +
       slot), using the Section 3 rules.
    2. Work out what the far player **should actually** do: play their newest
       not-already-known playable card; or, if they have none, discard their newest
       junk. That's the **target** instruction (kind + slot).
    3. **Kind:** if the tentative kind already matches the target kind, you'll
       **play**; if they differ, you'll **discard** (that's the flip).
    4. **Slot:** count how many slots *newer* the target card is than the tentative
       one (wrapping past slot 1 around to the oldest) — call it the *gap*. Act on
       **your own** slot number `1 + gap`.
    5. The card you land on must itself be safe to do (playable if you're playing,
       junk if discarding) — the giver guaranteed it is, so trust that.
- **You're the far player** (someone two seats back clued you) → wait to see the
  in-between player's move, then combine it with the clue to get your instruction.

#### Two easy ways to accidentally cause a bomb

- Giving a reactive clue when the in-between player has **no safe play/discard** to
  relay it with — their forced move then bombs.
- **Re-cluing a card the far player already knows (or is already going to play).**
  That produces a weird, often unsafe forced reaction. Don't re-clue settled cards.

### 5. What the bot does on its turn (priority order)

Each turn the bot takes the first of these that applies:

1. **React**, if the previous player just gave a reactive clue it must relay.
2. **Play** a card it knows — or can prove by card-counting — is playable.
3. **Give a reactive clue** that creates a play, if a safe one exists.
4. **Give a stable play clue** to its next neighbor.
5. **Discard** a card marked as junk (preferring ones it can prove are dead).
6. **Give a "this is junk" clue**, else discard its oldest unknown card.

(The endgame reshuffles these — next.)

### 6. The endgame

Once the deck is nearly out — specifically, once **the cards left to draw are no
more than the cards still needed for a perfect game** — the bot switches to endgame
mode:

- **It front-loads anything that scores** — plays, and clues that create plays,
  ahead of discarding.
- **It uses stalls.** A **number-4 or number-5 clue to a 4/5 that wasn't already
  known to be that rank is a "stall"**: it means *nothing actionable* — it just
  burns a clue so the team doesn't have to discard (and lose a card) while waiting
  for a play to line up. So in the endgame, **a number-5 clue is no longer a
  discard signal** — it just means "I'm passing."
- **It gambles on the last turn.** On its final turn, if a bot can *prove* (by
  elimination, since the deck is empty) that it holds a needed card no teammate can
  still play, it guesses and plays it — there's no downside once the game is
  ending.

### 7. One-page cheat sheet

```
SLOTS:  newest = slot 1 (you draw onto the LEFT); older cards = higher numbers.

CLUE THE NEXT PLAYER  (stable — a direct signal):
    color   -> play the newest card it touches
    number  -> discard the card just LEFT (newer) of the clued one
    number 1-> play ALL the 1s
    (endgame) number 4/5 onto a NEW 4/5 -> stall (means "pass")

CLUE THE FAR PLAYER  (reactive = the clue + the in-between player's move):
    in-between PLAYS    -> keep the kind  (play stays play)
    in-between DISCARDS -> flip the kind  (play <-> discard)
    in-between acts on slot 1 -> no shift; each slot older -> target moves 1 newer
    GOLDEN RULE: only give one if the in-between player has a SAFE play/discard to
                 relay it with — otherwise they bomb.

YOUR TURN:  play a play-signalled card  >  discard a discard-signalled card.
```

---

## Project layout

| Module | Purpose |
| --- | --- |
| `hanabi_sim/cards.py` | `Color`, `Card`, deck construction |
| `hanabi_sim/actions.py` | action types + the game-log record |
| `hanabi_sim/game.py` | the rules engine |
| `hanabi_sim/observation.py` | per-player view (own hand hidden) + inference helpers |
| `hanabi_sim/players/` | the strategies, including the reactor line |
| `hanabi_sim/runner.py` | run games, aggregate stats, parallel benchmarking |
| `hanabi_sim/recorder.py` + `viewer.py` | record a game, render the HTML replay |
| `hanabi_sim/cli.py` | the `python -m hanabi_sim` entry point |
| `hanabi_sim/live/` | the hanab.live client (`state.py` adapter, `client.py` websocket) |
| `run_live.py` | launch a bot on hanab.live |

## Rules implemented

Standard Hanabi: 5 suits × ranks `[1×3, 2×2, 3×2, 4×2, 5×1]` = 50 cards; hand size
5 (2–3 players) or 4 (4–5); 8 clue tokens; 3 strikes; completing a stack (playing a
5) refunds a clue; the game ends on the 3rd strike, on a perfect 25, or one full
lap after the deck empties.
