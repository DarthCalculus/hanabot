"""The *Reactor* bot: a reactive-clue convention (3 players).

This is a ground-up convention, distinct from the H-group-style ladder
(`playclue` ... `critsave`). Cards carry *signals* (play / discard) rather than
relying on chop. There are two kinds of clue:

* **Stable clue** -- to the *next* player (Bob). The clue's "initial signal" is
  applied directly to Bob's hand.
* **Reactive clue** -- to the player *after* next (Cathy, i.e. the one whose turn
  does not come next). It carries an *initial signal* (from the clue mechanics on
  Cathy's hand) which Bob then *transforms* into the *expected signal* by his own
  real turn: Bob **plays** = no flip, Bob **discards** = flip (play<->discard);
  the slot Bob acts on (``1+k``) **slides** the signal ``k`` slots left. Cathy
  decodes ``initial + Bob's reaction`` to learn which signal lands on her hand.

Slots: slot 1 = newest (leftmost; cards are drawn to the left). "To the left" =
toward newer = lower slot number, with wraparound at slot 1 -> the oldest slot.

Everything here is common knowledge: signals are derived by replaying the public
action log (touched orders, slot positions, clue-info history, and each reactive
clue's reaction = the immediately following action). No hidden information is
needed, so a player can derive the signals on their own hand too.

Faithful to the spec the user provided; scoped to 3 players. The one place worth
flagging is flowchart step 6 ("discard leftmost untouched card") -- "leftmost" =
newest here, which is unusual; it's behind ``DISCARD_NEWEST_FIRST`` so it's a
one-line flip if that turns out to be a typo for "chop" (oldest).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..actions import Action, ActionType
from ..cards import RANK_COUNTS, Card
from ..game import HAND_SIZE
from ..observation import Observation
from .base import Player


@dataclass
class _Derived:
    """Common-knowledge state reconstructed by replaying the log."""

    sig_play: set[int] = field(default_factory=set)
    sig_disc: set[int] = field(default_factory=set)
    clued: set[int] = field(default_factory=set)
    color_known: dict = field(default_factory=dict)   # order -> Color
    rank_known: dict = field(default_factory=dict)    # order -> rank
    # A trailing reactive clue with no reaction yet (giver, target, initial).
    pending: tuple | None = None


class ReactorPlayer(Player):
    name = "reactor"

    #: Step 6 default discard. True = "leftmost" (newest) per the doc's wording;
    #: flip to False to discard the oldest (chop) instead.
    DISCARD_NEWEST_FIRST = True

    #: Experiment: when True, a stable play-clue outranks discarding a
    #: discard-signalled card while >=2 clue tokens remain.
    STABLE_PLAY_BEFORE_DISCARD_AT_2 = False

    #: When True, assume teammates run the same card-counting self-deduction, so
    #: don't stable-clue a card a teammate can already prove playable, and
    #: deprioritise reactions that make them play such a card. Only sound when
    #: teammates actually self-deduce (the rdeduce family).
    ASSUME_TEAMMATES_DEDUCE = False

    #: Endgame convention (deck <= ENDGAME_DECK): a rank clue whose rank is in
    #: STALL_RANKS and that touches a card not previously clued with that rank is
    #: a STALL -- it applies no signal and is not a reaction. Gated to variants
    #: that use it (ReactorEndgamePlayer); STALL_RANKS lets variants add e.g. 4s.
    ENDGAME_FIVE_STALL = False
    ENDGAME_DECK = 5
    #: If set, trigger the endgame by hanab.live PACE (<= this value) instead of
    #: deck size. pace = deck + num_players - (max_score - score). Common
    #: knowledge, so giver and receiver agree.
    ENDGAME_PACE = None
    STALL_RANKS = (5,)
    #: On a player's last turn (deck empty), if their exact hand (deducible by
    #: elimination) provably contains a playable card, gamble: play the slot most
    #: likely to be it. Gated so a wrong guess can't strike out.
    LAST_TURN_GAMBLE = False
    #: Endgame convention: a color clue that fills in the color of a card already
    #: known (by rank) to be a 4 or 5 but not yet color-clued is also a STALL.
    COLOR_FILL_STALL = False
    #: Experiment: when no normal expected signal exists, allow a reactive clue to
    #: target Cathy's leftmost card that is *one* away from playable (rank ==
    #: stack+2), provided the giver can set the reaction up so Bob plays the
    #: unblocking card (rank stack+1) -- which makes Cathy's card playable by her
    #: turn. If no such reaction exists, no reactive clue is given.
    REACTIVE_ONE_AWAY = False

    #: Experiment (NOT adopted -- A/B within noise): when choosing among our own
    #: plays, only let a DEDUCED play outrank our signalled plays if no signalled
    #: card in our own hand could be a duplicate of it. If one might be, play the
    #: signalled (unknown) card first -- we can still re-verify the deduced (known)
    #: card next turn, but not vice versa (we'd blind-bomb the stale signal).
    #: Logically sound, but A/B (20k) 77.26% -> 77.36% perfect (within ~0.3pp
    #: noise) and +0.01pp strikeout: the dup strikes it avoids are mostly harmless.
    DEDUCE_DEFER_DUP_SIGNAL = False

    #: Experiment (NOT adopted -- A/B neutral-to-negative): giver-side good-touch,
    #: never create a play-signal on a card whose duplicate the holder can ALREADY
    #: deduce-play on their own. It targets the dominant non-gamble strike cause,
    #: but A/B (20k) was 77.26% -> 76.94% perfect: those duplicate strikes are
    #: mostly harmless (the game is usually still won) and refusing the clue loses
    #: more tempo than the rare strike costs. Giver-side only / sound; left off.
    GOODTOUCH_DEDUCIBLE_DUP = False

    # ==================================================================
    #  Turn entry point
    # ==================================================================
    def act(self, obs: Observation) -> Action:
        r = self._derive(obs)
        me = obs.player_index
        n = obs.num_players

        # 0) A reaction owed this turn is forced and overrides the flowchart.
        if r.pending is not None and n == 3:
            giver, _target, _I = r.pending
            if (giver + 1) % n == me:
                react = self._react(obs, r, r.pending)
                if react is not None:
                    return react

        # (subclass hook, priority #2) play a card deduced to be playable.
        extra = self._extra_play(obs, r)
        if extra is not None:
            return extra

        # 1) Play the leftmost card carrying a play signal.
        play = self._play_signaled(obs, r)
        if play is not None:
            return play

        # 2) Give a reactive clue if one is available.
        if obs.clue_tokens > 0 and n == 3:
            rc = self._reactive_clue(obs, r)
            if rc is not None:
                return rc

        # 3 & 4) Discard a discard-signalled card, and give a stable play clue.
        # Normally discard first; the experiment flag promotes the stable play
        # clue above the discard while >=2 clue tokens are available.
        def discard_signalled_step():
            if obs.clue_tokens < obs.max_clue_tokens:
                return self._discard_signaled(obs, r)
            return None

        def stable_play_step():
            if obs.clue_tokens > 0:
                return self._stable_clue(obs, r, want_play=True)
            return None

        if self.STABLE_PLAY_BEFORE_DISCARD_AT_2 and obs.clue_tokens >= 2:
            ordered_steps = (stable_play_step, discard_signalled_step)
        else:
            ordered_steps = (discard_signalled_step, stable_play_step)
        for step in ordered_steps:
            res = step()
            if res is not None:
                return res

        # 5) Stable clue that applies a discard signal to trash/duped card.
        if obs.clue_tokens > 0:
            sd = self._stable_clue(obs, r, want_play=False)
            if sd is not None:
                return sd

        # 6) Discard the leftmost untouched card.
        if obs.clue_tokens < obs.max_clue_tokens:
            d6 = self._discard_chop(obs, r)
            if d6 is not None:
                return d6

        # 7) Forced (8 clues): an 8-clue stall clue.
        if obs.clue_tokens > 0:
            st = self._stall_clue(obs)
            if st is not None:
                return st

        return self._fallback(obs, r)

    def _extra_play(self, obs: Observation, r: _Derived) -> Action | None:
        """Hook for an extra high-priority play (above signalled plays). None by
        default; subclasses (e.g. card-counting deduction) override it."""
        return None

    # ==================================================================
    #  Slot helpers (slot 1 = newest = highest order)
    # ==================================================================
    @staticmethod
    def _hand_slots(orders) -> tuple[dict, dict]:
        ordered = sorted(orders, reverse=True)  # newest (max order) first
        slots = {o: i + 1 for i, o in enumerate(ordered)}
        by_slot = {i + 1: o for i, o in enumerate(ordered)}
        return slots, by_slot

    @staticmethod
    def _point_left(s: int, by_slot: dict, handsize: int, touched_now: set) -> int:
        """First untouched slot stepping left (newer) from ``s`` with wraparound.

        Already-touched cards are skipped. If every card is touched, returns
        ``s`` itself (discard the touched card).
        """
        for step in range(1, handsize):
            cur = ((s - 1 - step) % handsize) + 1
            if by_slot[cur] not in touched_now:
                return cur
        return s

    def _engine_index_at_slot(self, obs: Observation, player: int, slot: int):
        hand = obs.hands[player]
        _slots, by_slot = self._hand_slots([cv.order for cv in hand])
        order = by_slot.get(slot)
        if order is None:
            return None
        for i, cv in enumerate(hand):
            if cv.order == order:
                return i
        return None

    # ==================================================================
    #  Initial-signal mechanics (shared by stable and reactive clues)
    # ==================================================================
    def _compute_initial(self, orders, touched, clued_before, color_known,
                         rank_known, is_color) -> tuple | None:
        """The (action, slot) a clue tells its *target*, before any reaction."""
        slots, by_slot = self._hand_slots(orders)
        handsize = len(orders)
        touched = [o for o in touched if o in slots]
        if not touched:
            return None
        new = [o for o in touched if o not in clued_before]

        if not new:
            # Re-clue (touches only already-clued cards). Act on the leftmost.
            o = min(touched, key=lambda x: slots[x])
            had_c = o in color_known
            had_r = o in rank_known
            new_info = (not had_c) if is_color else (not had_r)
            if new_info:
                return ("play", slots[o])
            if not (had_c and had_r):
                return ("discard", slots[o])
            return ("play" if is_color else "discard", slots[o])

        if is_color:
            return ("play", min(slots[o] for o in new))

        # Rank clue: each new card points to the first untouched card on its
        # left; discard the leftmost pointed-to card.
        touched_now = set(clued_before) | set(touched)
        ptrs = [self._point_left(slots[o], by_slot, handsize, touched_now) for o in new]
        return ("discard", min(ptrs))

    def _stable_signal_orders(self, orders, touched, clued_before, color_known,
                              rank_known, is_color, rank, clue_tokens, max_tok,
                              sig_play, stacks, colors) -> list[tuple[int, str]]:
        """(order, action) signals a *stable* clue imparts to the target."""
        new = [o for o in touched if o not in clued_before]

        # Rank-1: play-signal all the 1s, or discard-signal them all if they
        # can't all be played (stacks + 1s already carrying a play signal).
        if (not is_color) and rank == 1:
            needed = sum(1 for c in colors if stacks[c] == 0)
            already = sum(1 for o in sig_play if rank_known.get(o) == 1)
            play_all = len(new) <= (needed - already)
            return [(o, "play" if play_all else "discard") for o in new]

        # 8-clue stall: a 3/4/5 given at max tokens carries no signal.
        if (not is_color) and rank in (3, 4, 5) and clue_tokens == max_tok:
            return []

        I = self._compute_initial(orders, touched, clued_before, color_known,
                                  rank_known, is_color)
        if I is None:
            return []
        _slots, by_slot = self._hand_slots(orders)
        return [(by_slot[I[1]], I[0])]

    @staticmethod
    def _set_signal(sig_play: set, sig_disc: set, order: int, action: str) -> None:
        if action == "play":
            sig_play.add(order)
            sig_disc.discard(order)
        else:
            sig_disc.add(order)
            sig_play.discard(order)

    @staticmethod
    def _initial_hands(obs: Observation) -> list:
        """Reconstruct each player's STARTING hand (order ids, oldest-first) by
        rewinding the log from the current hands. Deal-order agnostic -- works
        whether the deal was round-robin (our engine) or sequential per player
        (hanab.live), so slot math is correct either way."""
        hands = [[cv.order for cv in obs.hands[p]] for p in range(obs.num_players)]
        for rec in reversed(obs.log):
            a = rec.action
            if a.type not in (ActionType.PLAY, ActionType.DISCARD):
                continue
            if a.card_index is None or rec.acted_order is None:
                continue
            h = hands[rec.player]
            if rec.drew_order is not None:          # undo the refill draw (newest)
                if h and h[-1] == rec.drew_order:
                    h.pop()
                elif rec.drew_order in h:
                    h.remove(rec.drew_order)
            h.insert(min(a.card_index, len(h)), rec.acted_order)  # undo the removal
        return hands

    # ==================================================================
    #  Log replay -> signals (common knowledge)
    # ==================================================================
    def _derive(self, obs: Observation) -> _Derived:
        n = obs.num_players
        max_tok = obs.max_clue_tokens
        H = HAND_SIZE[n]

        hands = self._initial_hands(obs)
        clued: set[int] = set()
        color_known: dict = {}
        rank_known: dict = {}
        sig_play: set[int] = set()
        sig_disc: set[int] = set()
        stacks = {c: 0 for c in obs.colors}
        clue_tokens = max_tok
        max_score = len(obs.colors) * 5
        deck = sum(RANK_COUNTS.values()) * len(obs.colors) - n * H  # cards left after the deal
        pending = None

        log = obs.log
        for i, rec in enumerate(log):
            a = rec.action
            if a.is_clue:
                giver = rec.player
                target = a.target
                is_color = a.type is ActionType.CLUE_COLOR
                touched = list(rec.touched_orders)
                clued_before = set(clued)
                torders = list(hands[target])

                in_zone = self._in_endgame(deck, sum(stacks.values()), n, max_score)
                stall_value = a.color if is_color else a.rank
                if self._stall_match(in_zone, is_color, stall_value, touched,
                                     rank_known, color_known):
                    pass  # endgame stall: no signal, no reaction
                elif n == 3 and target == (giver + 2) % n:
                    I = self._compute_initial(torders, touched, clued_before,
                                              color_known, rank_known, is_color)
                    self._apply_reactive(log, i, hands, n, torders, I,
                                         sig_play, sig_disc)
                    if i + 1 >= len(log):
                        pending = (giver, target, I)
                else:
                    sig = self._stable_signal_orders(
                        torders, touched, clued_before, color_known, rank_known,
                        is_color, a.rank, clue_tokens, max_tok, sig_play,
                        stacks, obs.colors)
                    for o, act in sig:
                        self._set_signal(sig_play, sig_disc, o, act)

                for o in touched:
                    if is_color:
                        color_known[o] = a.color
                    else:
                        rank_known[o] = a.rank
                clued.update(touched)
                clue_tokens = max(0, clue_tokens - 1)

            elif a.type in (ActionType.PLAY, ActionType.DISCARD):
                order = hands[rec.player][a.card_index]
                hands[rec.player].pop(a.card_index)
                if rec.drew_order is not None:
                    hands[rec.player].append(rec.drew_order)
                    deck -= 1  # a card was drawn from the deck
                sig_play.discard(order)
                sig_disc.discard(order)
                if a.type is ActionType.PLAY:
                    if rec.success:
                        stacks[rec.played_card.color] = rec.played_card.rank
                        if rec.played_card.rank == 5 and clue_tokens < max_tok:
                            clue_tokens += 1
                else:
                    clue_tokens = min(max_tok, clue_tokens + 1)

        current = {cv.order for hand in obs.hands for cv in hand}
        return _Derived(
            sig_play=sig_play & current,
            sig_disc=sig_disc & current,
            clued=clued,
            color_known=color_known,
            rank_known=rank_known,
            pending=pending,
        )

    def _apply_reactive(self, log, i, hands, n, torders, I,
                        sig_play, sig_disc) -> None:
        """Decode a reactive clue's reaction (log[i+1]) and apply Cathy's signal."""
        if I is None or i + 1 >= len(log):
            return
        giver = log[i].player
        bob = (giver + 1) % n
        nrec = log[i + 1]
        if nrec.player != bob or nrec.action.type not in (ActionType.PLAY, ActionType.DISCARD):
            return  # not a decodable reaction
        borders = list(hands[bob])
        ci = nrec.action.card_index
        if ci is None or not (0 <= ci < len(borders)):
            return
        bslots, _ = self._hand_slots(borders)
        k = bslots[borders[ci]] - 1
        flip = nrec.action.type is ActionType.DISCARD

        action = I[0]
        if flip:
            action = "discard" if action == "play" else "play"
        H = len(torders)
        final_slot = ((I[1] - 1 - k) % H) + 1  # the slide wraps around the hand
        _slots, by_slot = self._hand_slots(torders)
        self._set_signal(sig_play, sig_disc, by_slot[final_slot], action)

    # ==================================================================
    #  Expected signal / trash (need visible cards; never our own hand)
    # ==================================================================
    def _expected_signal(self, obs: Observation, who: int, r: _Derived) -> tuple | None:
        """Cathy's expected signal: (action, slot, order, intermediate), or None.

        ``intermediate`` is None except for the optional "1-away" case (see
        REACTIVE_ONE_AWAY): then it is the Card Bob must play to unblock Cathy's
        targeted card, and the giver must arrange for Bob's reaction to be exactly
        that play.

        CRUCIAL: this must be identical for Alice (giver) and Bob (reactor), who
        otherwise compute different slides and Bob plays an unverified card. So it
        uses only information common to both: the board, ``who``'s own hand, and
        order-based signals -- never the identity of a play-signaled card sitting
        in the observer's own (hidden) hand.
        """
        hand = obs.hands[who]
        slots, _ = self._hand_slots([cv.order for cv in hand])

        # Identities already signalled-to-play *within this hand* (both see them).
        psig_ids = {(cv.card.color, cv.card.rank) for cv in hand
                    if cv.order in r.sig_play and cv.card is not None}

        best = None
        for cv in hand:
            if cv.card is None or not obs.is_playable(cv.card):
                continue
            if cv.order in r.sig_play:
                continue
            if (cv.card.color, cv.card.rank) in psig_ids:
                continue
            s = slots[cv.order]
            if best is None or s < best[1]:
                best = ("play", s, cv.order, None)
        if best is not None:
            return best

        trash = None
        for cv in hand:
            if cv.card is None:
                continue
            dead = obs.is_dead(cv.card)
            dup_in_hand = any(o.order != cv.order and o.card == cv.card for o in hand)
            if not (dead or dup_in_hand):
                continue
            s = slots[cv.order]
            if trash is None or s < trash[1]:
                trash = ("discard", s, cv.order, None)
        if trash is not None:
            return trash

        # Last resort (opt-in): the leftmost card that is one play away from being
        # playable. Cathy will play it on her turn -- which only works if Bob's
        # reaction is to play the unblocking card first (enforced by the giver in
        # _reactive_clue). ``intermediate`` is that unblocking card.
        if self.REACTIVE_ONE_AWAY:
            near = None
            for cv in hand:
                if cv.card is None:
                    continue
                if cv.card.rank != obs.play_stacks[cv.card.color] + 2:
                    continue  # not exactly one away from playable
                if cv.order in r.sig_play:
                    continue
                if (cv.card.color, cv.card.rank) in psig_ids:
                    continue
                s = slots[cv.order]
                inter = Card(cv.card.color, cv.card.rank - 1)
                if near is None or s < near[1]:
                    near = ("play", s, cv.order, inter)
            if near is not None:
                return near
        return None

    @staticmethod
    def _is_trash(obs: Observation, order: int, card, r: _Derived) -> bool:
        if obs.is_dead(card):
            return True
        for hh in obs.hands:               # a visible duplicate -> safe to dump one
            for cv in hh:
                if cv.order != order and cv.card is not None and cv.card == card:
                    return True
        return False

    def _in_endgame(self, deck: int, score: int, num_players: int, max_score: int) -> bool:
        """Whether we're in the endgame zone -- by pace if ENDGAME_PACE is set,
        else by deck size. pace = deck + num_players - (max_score - score)."""
        if self.ENDGAME_PACE is not None:
            return deck + num_players - (max_score - score) <= self.ENDGAME_PACE
        return deck <= self.ENDGAME_DECK

    def _stall_match(self, in_zone: bool, is_color: bool, value,
                     touched, rank_known: dict, color_known: dict) -> bool:
        """True if a clue reads as an endgame stall (no signal / no reaction):
        - a rank clue whose rank is in STALL_RANKS, touching a card not yet clued
          with that rank; or
        - (COLOR_FILL_STALL) a color clue filling the color of a card already known
          by rank to be a 4 or 5 and not yet color-clued.
        All inputs are common knowledge, so giver and receiver always agree."""
        if not (self.ENDGAME_FIVE_STALL and in_zone):
            return False
        if is_color:
            if not self.COLOR_FILL_STALL:
                return False
            return any(rank_known.get(o) in (4, 5) and o not in color_known for o in touched)
        if value not in self.STALL_RANKS:
            return False
        return any(rank_known.get(o) != value for o in touched)

    def _dup_will_play(self, obs: Observation, r: _Derived, order: int, card) -> bool:
        for hh in obs.hands:
            for cv in hh:
                if cv.order != order and cv.card == card and cv.order in r.sig_play:
                    return True
        return False

    @staticmethod
    def _discard_priority(obs: Observation, card) -> int:
        """Rank a (safe-to-discard) card: 0 = actual trash (dead), 1 = live
        duplicate. Prefer dumping dead cards first so a redundant copy is kept
        as a backup."""
        return 0 if obs.is_dead(card) else 1

    # ==================================================================
    #  Reacting (we are Bob; play/discard blind, trusting Alice)
    # ==================================================================
    def _react(self, obs: Observation, r: _Derived, pending: tuple) -> Action | None:
        _giver, target, I = pending
        if I is None:
            return None
        E = self._expected_signal(obs, target, r)
        if E is None:
            return None
        flip = (I[0] != E[0])
        k = (I[1] - E[1]) % len(obs.hands[target])  # the slide wraps around the hand
        idx = self._engine_index_at_slot(obs, obs.player_index, 1 + k)
        if idx is None:
            return None
        if flip:
            if obs.clue_tokens >= obs.max_clue_tokens:
                return None
            return Action.discard(idx)
        return Action.play(idx)

    # ==================================================================
    #  Giving a reactive clue (we are Alice)
    # ==================================================================
    def _reactive_clue(self, obs: Observation, r: _Derived,
                       want: str | None = None) -> Action | None:
        """Best available reactive clue, or None. ``want`` filters by category:
        "bob_play" (Bob's reaction plays), "cathy_play_bob_discard" (Bob discards,
        Cathy plays), "two_discards" (Bob discards, Cathy discards). None = any.
        """
        me = obs.player_index
        n = obs.num_players
        cathy = (me + 2) % n
        bob = (me + 1) % n

        E = self._expected_signal(obs, cathy, r)
        if E is None:
            return None
        e_card = next((cv.card for cv in obs.hands[cathy] if cv.order == E[2]), None)

        chand = obs.hands[cathy]
        corders = [cv.order for cv in chand]

        # Giver-side good-touch: if Cathy can already deduce-play a copy of the
        # card this clue would signal, don't signal it (she'd play the deducible
        # copy and strike on this one). Alice-only filter; never touches decoding.
        if self.GOODTOUCH_DEDUCIBLE_DUP and self.ASSUME_TEAMMATES_DEDUCE \
                and e_card is not None:
            cathy_deduced_ids = {
                cv.card for cv in chand
                if cv.order in self._other_deducible_plays(obs, r, cathy)
            }
            if e_card in cathy_deduced_ids:
                return None
        bhand = obs.hands[bob]
        bslots, bby = self._hand_slots([cv.order for cv in bhand])
        bcard = {cv.order: cv.card for cv in bhand}
        bob_handsize = len(bhand)

        leftmost_nopsig = min(
            (bslots[cv.order] for cv in bhand if cv.order not in r.sig_play),
            default=99)

        # Cards already signalled to play in Bob's or Cathy's hand (order +
        # identity). Bob must never react by PLAYING a duplicate of one of these:
        # finishing his copy would strand the already-signalled copy as a dead
        # card (the cause of the seed-10237 strikeouts).
        signalled_plays = [
            (cv.order, (cv.card.color, cv.card.rank))
            for hand in (bhand, chand) for cv in hand
            if cv.order in r.sig_play and cv.card is not None
        ]

        # Bob's cards he can already prove playable himself: prefer NOT spending
        # his reaction on one of these (he'd play it on his own anyway).
        bob_deducible = (self._other_deducible_plays(obs, r, bob)
                         if self.ASSUME_TEAMMATES_DEDUCE else set())
        in_zone = self._in_endgame(obs.deck_size, sum(obs.play_stacks.values()),
                                   obs.num_players, len(obs.colors) * 5)

        options = [(True, c, None) for c in {cv.card.color for cv in chand}]
        options += [(False, None, rk) for rk in {cv.card.rank for cv in chand}]

        candidates: list[tuple[tuple, Action]] = []
        for is_color, color, rank in options:
            touched = [cv.order for cv in chand
                       if (cv.card.color == color if is_color else cv.card.rank == rank)]
            # A clue that reads as an endgame stall can't carry a signal.
            if self._stall_match(in_zone, is_color, color if is_color else rank,
                                 touched, r.rank_known, r.color_known):
                continue
            I = self._compute_initial(corders, touched, r.clued,
                                      r.color_known, r.rank_known, is_color)
            if I is None:
                continue
            flip = (I[0] != E[0])
            k = (I[1] - E[1]) % len(chand)  # the slide wraps around the hand
            bslot = 1 + k
            if bslot > bob_handsize:
                continue
            order = bby[bslot]
            card = bcard[order]
            # A play-reaction must not finish a duplicate of a card already
            # signalled to play elsewhere (it would kill the signalled copy).
            if not flip and any(o != order and ident == (card.color, card.rank)
                                for o, ident in signalled_plays):
                continue
            intermediate = E[3]
            if intermediate is not None:
                # 1-away: Bob's reaction must be to PLAY exactly the unblocking
                # card, so Cathy's targeted card is playable by her turn.
                if flip or card != intermediate:
                    continue
            elif not flip:
                if not obs.is_playable(card):
                    continue
                # Bob's reaction-play must not be the same card Cathy will play
                # (it would advance the stack and kill her signalled card).
                if e_card is not None and card == e_card:
                    continue
            else:
                if obs.clue_tokens - 1 >= obs.max_clue_tokens:
                    continue  # Bob couldn't discard
                if not self._is_trash(obs, order, card, r):
                    continue

            if want is not None:
                if want == "bob_play" and flip:
                    continue
                if want == "cathy_play_bob_discard" and not (flip and E[0] == "play"):
                    continue
                if want == "two_discards" and not (flip and E[0] == "discard"):
                    continue

            # For discard reactions, prefer dumping actual trash over a live dup.
            dprio = self._discard_priority(obs, card) if flip else 0
            # A play-reaction on a card Bob can self-deduce is redundant (he'd
            # play it himself) -- still valid, but sorted after other plays.
            redundant = (not flip) and (order in bob_deducible)
            key = self._reactive_key(obs, flip, bslot, k, leftmost_nopsig, dprio,
                                     redundant)
            act = (Action.clue_color(cathy, color) if is_color
                   else Action.clue_rank(cathy, rank))
            candidates.append((key, act))

        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    @staticmethod
    def _reactive_key(obs, flip, bslot, k, leftmost_nopsig, dprio, redundant=False) -> tuple:
        """Lower sorts first. Plays are preferred at >=2 clues, discards at <2;
        among plays, a non-redundant card (Bob can't self-deduce it) beats a
        redundant one, then the leftmost no-play-signal slot; among discards,
        actual trash (dprio 0) beats a duplicate (1), then the leftmost slot."""
        is_play = not flip
        if is_play:
            play_key = (1 if redundant else 0,
                        0 if bslot == leftmost_nopsig else 1, bslot, k)
            return (0, play_key) if obs.clue_tokens >= 2 else (1, play_key)
        disc_key = (dprio, bslot, k)
        return (1, disc_key) if obs.clue_tokens >= 2 else (0, disc_key)

    # ==================================================================
    #  Giving a stable clue (we are Alice, target = Bob)
    # ==================================================================
    def _stable_clue(self, obs: Observation, r: _Derived, want_play: bool) -> Action | None:
        me = obs.player_index
        n = obs.num_players
        bob = (me + 1) % n
        if bob == me:
            return None
        bhand = obs.hands[bob]
        borders = [cv.order for cv in bhand]
        cardof = {cv.order: cv.card for cv in bhand}
        slots, _ = self._hand_slots(borders)

        # Cards Bob can already prove playable himself -- no need to clue them.
        deducible = (self._other_deducible_plays(obs, r, bob)
                     if (want_play and self.ASSUME_TEAMMATES_DEDUCE) else set())
        in_zone = self._in_endgame(obs.deck_size, sum(obs.play_stacks.values()),
                                   obs.num_players, len(obs.colors) * 5)

        options = [(True, c, None) for c in {cv.card.color for cv in bhand}]
        options += [(False, None, rk) for rk in {cv.card.rank for cv in bhand}]

        best = None
        for is_color, color, rank in options:
            touched = [cv.order for cv in bhand
                       if (cv.card.color == color if is_color else cv.card.rank == rank)]
            # A clue that reads as an endgame stall can't carry a signal.
            if self._stall_match(in_zone, is_color, color if is_color else rank,
                                 touched, r.rank_known, r.color_known):
                continue
            sig = self._stable_signal_orders(
                borders, touched, r.clued, r.color_known, r.rank_known,
                is_color, rank, obs.clue_tokens, obs.max_clue_tokens,
                r.sig_play, obs.play_stacks, obs.colors)
            if not sig:
                continue

            if want_play:
                plays = [o for o, act in sig if act == "play"]
                if not plays:
                    continue
                if any(not obs.is_playable(cardof[o]) for o in plays):
                    continue
                if any(self._dup_will_play(obs, r, o, cardof[o]) for o in plays):
                    continue
                # No good-touch violation: never signal two copies of one card.
                ids = [cardof[o] for o in plays]
                if len(set(ids)) < len(ids):
                    continue
                # Skip if Bob can already prove this card playable on his own.
                if any(o in deducible for o in plays):
                    continue
                # ...or if a DUPLICATE of it is deducible in Bob's hand: he'd
                # play that copy and strike on this one later.
                if self.GOODTOUCH_DEDUCIBLE_DUP:
                    deduced_ids = {cardof[o2] for o2 in deducible}
                    if any(cardof[o] in deduced_ids for o in plays):
                        continue
                key = (min(slots[o] for o in plays), min(cardof[o].rank for o in plays))
            else:
                discs = [o for o, act in sig if act == "discard"]
                if not discs:
                    continue
                if any(not self._is_trash(obs, o, cardof[o], r) for o in discs):
                    continue
                # Actual trash before a live duplicate, then leftmost slot.
                key = (min(self._discard_priority(obs, cardof[o]) for o in discs),
                       min(slots[o] for o in discs))

            act = (Action.clue_color(bob, color) if is_color
                   else Action.clue_rank(bob, rank))
            if best is None or key < best[0]:
                best = (key, act)

        return None if best is None else best[1]

    # ==================================================================
    #  Plays / discards driven by our own signals
    # ==================================================================
    def _play_signaled(self, obs: Observation, r: _Derived) -> Action | None:
        own = obs.own_hand
        slots, _ = self._hand_slots([cv.order for cv in own])
        elsewhere = None
        best = None
        for i, cv in enumerate(own):
            if cv.order not in r.sig_play:
                continue
            # Skip a signalled card we can now PROVE won't play (a stale signal
            # whose stack advanced past it) -- otherwise it would bomb.
            if elsewhere is None:
                elsewhere = self._copies_elsewhere(obs)
            rem = self._remaining_identities(obs, i, elsewhere)
            if rem and not any(rank == obs.play_stacks[c] + 1 for c, rank in rem):
                continue
            s = slots[cv.order]
            if best is None or s < best[0]:
                best = (s, i)
        return None if best is None else Action.play(best[1])

    # --- card-counting deduction of a hidden own card ---------------------
    @staticmethod
    def _copies_excluding(obs: Observation, exclude) -> dict:
        """Copies of each (color, rank) on the stacks + discard + the hands of
        players NOT in ``exclude``."""
        counts: dict = {}
        for c in obs.colors:
            for rank in range(1, obs.play_stacks[c] + 1):  # one of each played rank
                counts[(c, rank)] = counts.get((c, rank), 0) + 1
        for card in obs.discard_pile:
            counts[(card.color, card.rank)] = counts.get((card.color, card.rank), 0) + 1
        for p in range(obs.num_players):
            if p in exclude:
                continue
            for cv in obs.hands[p]:
                if cv.card is not None:
                    key = (cv.card.color, cv.card.rank)
                    counts[key] = counts.get(key, 0) + 1
        return counts

    def _copies_elsewhere(self, obs: Observation) -> dict:
        """Copies visible *outside our own hand* -- what our own deduction uses."""
        return self._copies_excluding(obs, (obs.player_index,))

    def _other_deducible_plays(self, obs: Observation, r: _Derived, other: int) -> set:
        """Orders in ``other``'s hand that we're SURE ``other`` can prove playable
        on their own (and would play). Conservative: counts only copies we're
        certain ``other`` sees -- excluding both their hand and our own (hidden)
        hand -- so this never over-claims a card they can't actually deduce."""
        me = obs.player_index
        counts = self._copies_excluding(obs, (other, me))
        sig_ids = {
            (cv.card.color, cv.card.rank)
            for p in range(obs.num_players) if p not in (other, me)
            for cv in obs.hands[p]
            if cv.order in r.sig_play and cv.card is not None
        }
        out: set = set()
        for cv in obs.hands[other]:
            remaining = [
                (c, rank)
                for c in cv.possible_colors for rank in cv.possible_ranks
                if counts.get((c, rank), 0) < RANK_COUNTS[rank]
            ]
            if (remaining
                    and all(rank == obs.play_stacks[c] + 1 for c, rank in remaining)
                    and not any(ident in sig_ids for ident in remaining)):
                out.add(cv.order)
        return out

    @staticmethod
    def _remaining_identities(obs: Observation, i: int, elsewhere: dict) -> list:
        """Identities own-hand card ``i`` could still be: its clue possibilities
        minus any identity whose every copy is already accounted for elsewhere.
        The card's true identity is always retained (its own copy isn't counted),
        so this is sound -- it never eliminates the real card."""
        cv = obs.own_hand[i]
        return [(c, rank)
                for c in cv.possible_colors for rank in cv.possible_ranks
                if elsewhere.get((c, rank), 0) < RANK_COUNTS[rank]]

    def _last_turn_gamble(self, obs: Observation) -> Action | None:
        """On a final-round turn (deck empty), if our exact hand provably holds a
        playable card that no still-to-act teammate also holds, gamble on the slot
        most likely to be it. Worth it even at risk of a 3rd strike: the game is
        ending, so a wrong guess costs no future points -- and the point is ours
        to grab or lose (no later teammate holds it)."""
        if obs.deck_size > 0:
            return None
        # exact own hand by elimination -- deck empty, so all else is visible
        my = {(c, r): cnt for c in obs.colors for r, cnt in RANK_COUNTS.items()}

        def dec(c, r):
            if my.get((c, r), 0) > 0:
                my[(c, r)] -= 1

        for p in range(obs.num_players):
            if p == obs.player_index:
                continue
            for cv in obs.hands[p]:
                if cv.card is not None:
                    dec(cv.card.color, cv.card.rank)
        for card in obs.discard_pile:
            dec(card.color, card.rank)
        for c in obs.colors:
            for r in range(1, obs.play_stacks[c] + 1):
                dec(c, r)
        my = {k: v for k, v in my.items() if v > 0}
        if sum(my.values()) != len(obs.own_hand):
            return None  # couldn't pin the hand exactly -- don't risk it

        # players who still act after us this final round (one lap after the
        # last draw); a playable they also hold will be covered by them.
        log = obs.log
        last_draw = max((i for i, rec in enumerate(log) if rec.drew_order is not None),
                        default=-1)
        turns_after_me = (last_draw + obs.num_players) - len(log)
        covered = set()
        for j in range(max(0, turns_after_me)):
            p = (obs.player_index + 1 + j) % obs.num_players
            for cv in obs.hands[p]:
                if cv.card is not None:
                    covered.add((cv.card.color, cv.card.rank))

        useful = {(c, r) for (c, r) in my
                  if r == obs.play_stacks[c] + 1 and (c, r) not in covered}
        if not useful:
            return None

        best = None
        for i, cv in enumerate(obs.own_hand):
            cands = [(c, r) for c in cv.possible_colors for r in cv.possible_ranks
                     if (c, r) in my]
            good = sum(1 for ident in cands if ident in useful)
            if not cands or good == 0:
                continue
            key = (good / len(cands), -len(cands))   # most likely useful-playable
            if best is None or key > best[0]:
                best = (key, i)
        return None if best is None else Action.play(best[1])

    def _trash_discard(self, obs: Observation) -> Action | None:
        """Discard a card we can PROVE is trash (every possible identity dead),
        if any -- ties broken by the usual discard ordering. Always safe: the
        card's true identity is among its remaining (sound) possibilities, which
        are all dead. Used whenever the bot chooses to discard."""
        elsewhere = self._copies_elsewhere(obs)
        slots, _ = self._hand_slots([cv.order for cv in obs.own_hand])
        idxs = [
            i for i, cv in enumerate(obs.own_hand)
            if (rem := self._remaining_identities(obs, i, elsewhere))
            and all(rank <= obs.play_stacks[c] for c, rank in rem)
        ]
        if not idxs:
            return None
        idxs.sort(key=lambda i: slots[obs.own_hand[i].order],
                  reverse=not self.DISCARD_NEWEST_FIRST)
        return Action.discard(idxs[0])

    def _discard_signaled(self, obs: Observation, r: _Derived) -> Action | None:
        own = obs.own_hand
        slots, _ = self._hand_slots([cv.order for cv in own])
        best = None
        for i, cv in enumerate(own):
            if cv.order in r.sig_disc:
                s = slots[cv.order]
                if best is None or s < best[0]:
                    best = (s, i)
        if best is None:
            return None  # trigger: only fires when a discard signal exists
        return self._trash_discard(obs) or Action.discard(best[1])

    def _discard_chop(self, obs: Observation, r: _Derived) -> Action | None:
        trash = self._trash_discard(obs)
        if trash is not None:
            return trash
        own = obs.own_hand
        slots, _ = self._hand_slots([cv.order for cv in own])
        cands = [(slots[cv.order], i) for i, cv in enumerate(own)
                 if (not cv.clued) and cv.order not in r.sig_play and cv.order not in r.sig_disc]
        if not cands:
            return None
        cands.sort(reverse=not self.DISCARD_NEWEST_FIRST)
        return Action.discard(cands[0][1])

    # ==================================================================
    #  Forced stalls / fallbacks
    # ==================================================================
    def _stall_clue(self, obs: Observation) -> Action | None:
        me = obs.player_index
        bob = (me + 1) % obs.num_players
        bhand = obs.hands[bob]
        for rank in (5, 4, 3):
            if any(cv.card.rank == rank for cv in bhand):
                return Action.clue_rank(bob, rank)
        return self._any_clue(obs)

    def _fallback(self, obs: Observation, r: _Derived) -> Action:
        if obs.clue_tokens < obs.max_clue_tokens:
            trash = self._trash_discard(obs)
            if trash is not None:
                return trash
            own = obs.own_hand
            slots, _ = self._hand_slots([cv.order for cv in own])
            cands = [(slots[cv.order], i) for i, cv in enumerate(own)
                     if cv.order not in r.sig_play]
            if cands:
                cands.sort()
                return Action.discard(cands[0][1])
            return Action.discard(len(own) - 1)
        clue = self._any_clue(obs)
        return clue if clue is not None else Action.play(0)

    def _any_clue(self, obs: Observation) -> Action | None:
        for p in obs.other_players():
            hand = obs.hands[p]
            if not hand:
                continue
            return Action.clue_rank(p, hand[0].card.rank)
        return None
