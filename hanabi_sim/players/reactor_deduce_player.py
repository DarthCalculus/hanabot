"""Reactor + card-counting self-deduction of playable cards.

On top of the Reactor convention, a player tries to *prove* one of their own
cards is playable. Starting from the card's clue knowledge (positive + negative),
it eliminates every identity all of whose copies are already accounted for
elsewhere -- on the stacks, in the discard pile, or in other players' hands. If
the remaining possibilities are all currently playable AND none of them is a card
already signalled to play in someone's hand (so playing it can't dupe a teammate's
play), the card is certainly safe to play.

This play sits at priority #2: a forced reaction still comes first, but a card
proven playable this way is played before anything else (above signalled plays
and all clue/discard options). The deduction is conservative -- it only ever
keeps extra possibilities (it can't see the player's own other cards), so it
never eliminates wrongly and never causes a misplay.
"""

from __future__ import annotations

from ..actions import Action
from ..observation import Observation
from .reactor_player import ReactorPlayer, _Derived


class ReactorDeducePlayer(ReactorPlayer):
    name = "rdeduce"

    #: Teammates run this same deduction, so skip cluing / deprioritise reacting
    #: with cards they can already prove playable themselves.
    ASSUME_TEAMMATES_DEDUCE = True

    def _extra_play(self, obs: Observation, r: _Derived) -> Action | None:
        stacks = obs.play_stacks
        elsewhere = self._copies_elsewhere(obs)
        signalled_ids = {
            (cv.card.color, cv.card.rank)
            for hand in obs.hands for cv in hand
            if cv.order in r.sig_play and cv.card is not None
        }

        own = obs.own_hand
        slots, _ = self._hand_slots([cv.order for cv in own])
        best = None
        for i, cv in enumerate(own):
            remaining = self._remaining_identities(obs, i, elsewhere)
            if not remaining:
                continue
            if not all(rank == stacks[c] + 1 for c, rank in remaining):
                continue  # not provably playable right now
            if any(ident in signalled_ids for ident in remaining):
                continue  # could be a copy of a card already going to be played
            # A signalled card in OUR OWN hand is hidden, so the check above can't
            # see it. If one might be a duplicate of this deduced card, defer the
            # deduced (known) play and let the signalled (unknown) one go first:
            # afterwards we can still re-verify this card and skip it if its stack
            # moved -- whereas the reverse order would blind-bomb the stale signal.
            if self.DEDUCE_DEFER_DUP_SIGNAL and any(
                j != i and own[j].order in r.sig_play
                and set(remaining) & set(self._remaining_identities(obs, j, elsewhere))
                for j in range(len(own))
            ):
                continue
            s = slots[cv.order]
            if best is None or s < best[0]:
                best = (s, i)
        return None if best is None else Action.play(best[1])


class ReactorEndgamePlayer(ReactorDeducePlayer):
    """rdeduce, but in the endgame (<= ENDGAME_DECK cards left) it front-loads
    play-generating clues over discards, per this flowchart:

      1. react if needed
      2. play (deduced, then signalled)
      3. reactive clue whose reaction makes Bob play
      4. stable clue that play-signals Bob
      5. reactive clue that makes Cathy play (Bob discards)
      5b. 5-stall: rank-5 clue to a not-yet-5-clued 5 (no signal)
      6. discard provable trash / a discard-signalled card
      7. reactive clue that makes both Bob and Cathy discard
      8. discard the chop
      9. stable clue that discard-signals trash
    """

    name = "rdend"
    ENDGAME_DECK = 6  # fallback only; superseded by ENDGAME_PACE
    #: Trigger the endgame by PACE <= 3 (== deck <= plays-remaining), which A/B'd
    #: better than any fixed deck size: +~1pp winrate (both seeds) vs deck<=6.
    ENDGAME_PACE = 3
    ENDGAME_FIVE_STALL = True
    #: Adopted: stall with 4s as well as 5s in the endgame. A/B (2 seeds, 4000):
    #: +~0.07 mean, +~3pp perfect vs 5-only. (Set to (5,) for 5-only.)
    STALL_RANKS = (5, 4)

    #: Adopted: place the 5-stall above the reactive clues where Bob discards
    #: (steps "cathy plays" and "two discards"), rather than just above the trash
    #: discard. A/B (2 seeds, 4000 games): +~0.085 mean, +~5pp perfect.
    FIVE_STALL_ABOVE_BOB_DISCARD = True

    #: Adopted: last-turn gamble for an otherwise-lost playable. A/B (20k): +0.8pp
    #: winrate, +0.028 mean, still 0% strikeout.
    LAST_TURN_GAMBLE = True

    #: Adopted: 1-away reactive clues -- when Cathy has no playable/trash, a
    #: reactive clue can target her leftmost card one play from playable, with
    #: Bob's reaction playing the unblocking card (two plays from one clue). A/B
    #: (20k, same seed): 75.83% -> 77.11% perfect (+1.28pp), +0.028 mean, +0.01pp
    #: strikeout (a rare stale 1-away signal).
    REACTIVE_ONE_AWAY = True

    #: Adopted: among 4-stalls, prefer ones that fill a still-useful (non-dead) 4
    #: over ones that only fill dead 4s. 5-stalls stay on top. A/B (2 seeds, 4000):
    #: +~0.2pp perfect, both seeds.
    PREFER_LIVE_FOUR_STALL = True

    #: Experiment: don't GIVE a stall whose newly-filled cards are all dead/trash
    #: (interpretation is unchanged -- such a clue is still read as a stall if it
    #: ever occurs -- the giver just won't choose it). Giver-side only / safe.
    SKIP_ALL_DEAD_STALL = False

    def _endgame_stall_clue(self, obs: Observation, r: _Derived) -> Action | None:
        """Clue a stall rank to a teammate holding such a card not yet clued with
        that rank, choosing by `_stall_tier`."""
        if not self.ENDGAME_FIVE_STALL or obs.clue_tokens <= 0:
            return None
        if self.LAST_TURN_GAMBLE and obs.deck_size == 0:
            return None  # last round: stalling is pointless; leave room to gamble
        if not self._in_endgame(obs.deck_size, sum(obs.play_stacks.values()),
                                obs.num_players, len(obs.colors) * 5):
            return None
        # When stalls are stable-only, only the next player may be stalled (a
        # stall to the far player would be misread as a reactive clue).
        me, nn = obs.player_index, obs.num_players
        stall_targets = ([(me + 1) % nn] if self.STALLS_STABLE_ONLY
                         else list(obs.other_players()))
        best = None  # (tier, action)
        for rank in self.STALL_RANKS:
            for p in stall_targets:
                new = [cv for cv in obs.hands[p]
                       if cv.card.rank == rank and r.rank_known.get(cv.order) != rank]
                if not new:
                    continue
                if self.SKIP_ALL_DEAD_STALL and all(obs.is_dead(cv.card) for cv in new):
                    continue  # a pure-trash stall conveys nothing -- don't bother
                tier = self._stall_tier(obs, rank, new)
                if best is None or tier < best[0]:
                    best = (tier, Action.clue_rank(p, rank))

        # Color-fill stalls: complete the colour of a known 4/5 not yet coloured.
        # Tier secondary key 2 slots them after rank-4 within each live/dead band:
        # 5, live-4, colour-live, dead-4, colour-trash.
        if self.COLOR_FILL_STALL:
            for p in obs.other_players():
                for c in {cv.card.color for cv in obs.hands[p]}:
                    fills = [cv for cv in obs.hands[p] if cv.card.color == c
                             and r.rank_known.get(cv.order) in (4, 5)
                             and cv.order not in r.color_known]
                    if not fills:
                        continue
                    live = any(not obs.is_dead(cv.card) for cv in fills)
                    tier = (0 if live else 1, 2)
                    if best is None or tier < best[0]:
                        best = (tier, Action.clue_color(p, c))
        return None if best is None else best[1]

    def _stall_tier(self, obs: Observation, rank: int, new_cards) -> tuple:
        """Lower = preferred. Default: higher rank first (5 before 4 before 3).
        With PREFER_LIVE_FOUR_STALL: stalls marking a still-useful (non-dead) card
        come first, then higher rank -- i.e. 5, live-4, live-3, then dead 4s/3s.
        (5s in hand are always live, so they stay on top.)"""
        if not self.PREFER_LIVE_FOUR_STALL:
            return (5 - rank,)
        live = any(not obs.is_dead(cv.card) for cv in new_cards)
        return (0 if live else 1, 5 - rank)

    def act(self, obs: Observation) -> Action:
        in_zone = self._in_endgame(obs.deck_size, sum(obs.play_stacks.values()),
                                   obs.num_players, len(obs.colors) * 5)
        if not in_zone:
            return super().act(obs)
        return self._endgame_act(obs)

    def _endgame_act(self, obs: Observation) -> Action:
        r = self._derive(obs)
        me, n = obs.player_index, obs.num_players
        clued = obs.clue_tokens > 0
        can_discard = obs.clue_tokens < obs.max_clue_tokens

        # 1. forced reaction
        if r.pending is not None and n == 3:
            giver, _t, _I = r.pending
            if (giver + 1) % n == me:
                react = self._react(obs, r, r.pending)
                if react is not None:
                    return react

        # 2. play (deduced, then signalled)
        for play in (self._extra_play(obs, r), self._play_signaled(obs, r)):
            if play is not None:
                return play

        # 3-9: play-generating clues first, discards/discard-clues last.
        bob_play = lambda: self._reactive_clue(obs, r, want="bob_play") if clued and n == 3 else None
        stable_play = lambda: self._stable_clue(obs, r, want_play=True) if clued else None
        cathy_play = lambda: self._reactive_clue(obs, r, want="cathy_play_bob_discard") if clued and n == 3 else None
        five_stall = lambda: self._endgame_stall_clue(obs, r) if clued else None
        trash_disc = lambda: (self._trash_discard(obs) or self._discard_signaled(obs, r)) if can_discard else None
        two_disc = lambda: self._reactive_clue(obs, r, want="two_discards") if clued and n == 3 else None
        chop = lambda: self._discard_chop(obs, r) if can_discard else None
        stable_disc = lambda: self._stable_clue(obs, r, want_play=False) if clued else None
        # Last-turn gamble: below sure plays / play-clues, above discards. Fires
        # only on the final round (deck empty); returns None otherwise.
        gamble = lambda: self._last_turn_gamble(obs) if self.LAST_TURN_GAMBLE else None

        if self.FIVE_STALL_ABOVE_BOB_DISCARD:
            # 5-stall outranks every reactive clue where Bob discards.
            steps = [bob_play, stable_play, five_stall, cathy_play,
                     gamble, trash_disc, two_disc, chop, stable_disc]
        else:
            steps = [bob_play, stable_play, cathy_play, five_stall,
                     gamble, trash_disc, two_disc, chop, stable_disc]
        for step in steps:
            res = step()
            if res is not None:
                return res

        # Forced fallback (e.g. 8 clues, nothing above fired).
        if clued:
            st = self._stall_clue(obs)
            if st is not None:
                return st
        return self._fallback(obs, r)


class ReactorScoredPlayer(ReactorEndgamePlayer):
    """Experiment: score every candidate clue (reactive = expected-signal score +
    reaction score; stable = its signal's score) and pick by score.

    Action value (same scale for a reaction, an expected signal, and a stable
    signal):
      PLAY  : +0 if already play-signalled / CK-deducibly playable / a dup of such;
              +1 if on a touched card or a dup of a touched card;
              +2 if on an untouched (novel) card.
      DISCARD: +0 if already discard-signalled or CK-dead trash;
               else +1 (>=4 clues), +1.5 (2-3 clues), +2 (0-1 clues).

    Flowchart (non-endgame): 1 react, 2 play, 3 best clue scoring >2, 4 discard a
    discard-signal/trash, 5 best clue scoring >0, 6 discard the chop.
    """

    name = "rdscore"
    EXPECTED_SKIP_SIGNALLED = True

    #: Flat value of a discard-signal (telling a holder to dump a non-obvious
    #: trash). Clue scarcity is NOT priced in here -- it lives in the threshold.
    DISCARD_CLUE_VALUE = 1.0

    #: Play-signal values. PLAY_NOVEL = a play on an untouched (newly-revealed)
    #: card; PLAY_TOUCHED = a play on an already-touched card or a dup of one.
    #: Trying PLAY_TOUCHED = 2.0 (don't distinguish touched plays); was 1.0.
    PLAY_NOVEL_VALUE = 2.0
    PLAY_TOUCHED_VALUE = 2.0

    #: Step-3 clue threshold by clue count: a clue is taken over discarding only
    #: if it scores strictly above this. (>=4, 2-3, 0-1 clues.) A rising schedule
    #: was swept and did NOT beat a flat 2.0, so it's left flat; kept parameterised
    #: in case it helps once other gaps close.
    THRESH_HI = 2.0
    THRESH_MID = 2.0
    THRESH_LO = 2.0

    def _step3_threshold(self, clue_tokens: int) -> float:
        if clue_tokens >= 4:
            return self.THRESH_HI
        if clue_tokens >= 2:
            return self.THRESH_MID
        return self.THRESH_LO

    # --- action scoring -------------------------------------------------
    def _play_value(self, obs, r, order, card, deducible, idmap) -> float:
        pid = (card.color, card.rank)
        if order in r.sig_play or order in deducible:
            return 0.0  # already going to be played
        known_play_ids = ({idmap[o] for o in r.sig_play if o in idmap}
                          | {idmap[o] for o in deducible if o in idmap})
        if pid in known_play_ids:
            return 0.0  # a dup of a card already known to be played
        if order in r.clued or pid in {idmap[o] for o in r.clued if o in idmap}:
            return self.PLAY_TOUCHED_VALUE  # touched, or a dup of a touched card
        return self.PLAY_NOVEL_VALUE  # untouched, novel

    def _discard_value(self, obs, r, order, card, deducible_trash) -> float:
        # Only trash / duplicates may be discard-signalled -- signalling a discard
        # of a genuinely useful card is not a legal convention move.
        if not self._is_trash(obs, order, card, r):
            return -100.0
        # +0 only when the discard is already known: already discard-signalled, or
        # the holder can PROVE it's trash from common knowledge. Otherwise a flat
        # value -- clue scarcity is handled by the step-3 threshold, NOT here.
        if order in r.sig_disc or order in deducible_trash:
            return 0.0
        return self.DISCARD_CLUE_VALUE

    @staticmethod
    def _clue_key(c):
        """Total, process-stable ordering: highest score first, reactive before
        stable on ties, then by target and clue value -- so selection never
        depends on (hash-randomised) set iteration order."""
        score, kind, act = c
        if act.color is not None:
            sub = (0, act.color.value, 0)
        else:
            sub = (1, "", act.rank or 0)
        return (-score, kind, act.target, sub)

    # --- scored clue enumeration ---------------------------------------
    def _clue_candidates(self, obs, r):
        me, n = obs.player_index, obs.num_players
        ct = obs.clue_tokens
        if ct <= 0:
            return []
        idmap = {cv.order: (cv.card.color, cv.card.rank)
                 for p in range(n) if p != me
                 for cv in obs.hands[p] if cv.card is not None}
        out = self._stable_candidates(obs, r, idmap, ct)
        if n == 3:
            out += self._reactive_candidates(obs, r, idmap, ct)
        return out

    def _stable_candidates(self, obs, r, idmap, ct):
        me, n = obs.player_index, obs.num_players
        bob = (me + 1) % n
        bhand = obs.hands[bob]
        borders = [cv.order for cv in bhand]
        cardof = {cv.order: cv.card for cv in bhand}
        deducible = self._other_deducible_plays(obs, r, bob)
        deducible_trash = self._other_deducible_trash(obs, r, bob)
        in_zone = self._in_endgame(obs.deck_size, sum(obs.play_stacks.values()),
                                   obs.num_players, len(obs.colors) * 5)
        options = [(True, c, None) for c in {cv.card.color for cv in bhand}]
        options += [(False, None, rk) for rk in {cv.card.rank for cv in bhand}]
        out = []
        for is_color, color, rank in options:
            touched = [cv.order for cv in bhand
                       if (cv.card.color == color if is_color else cv.card.rank == rank)]
            if self._stall_match(in_zone, is_color, color if is_color else rank,
                                 touched, r.rank_known, r.color_known):
                continue
            sig = self._stable_signal_orders(
                borders, touched, r.clued, r.color_known, r.rank_known,
                is_color, rank, ct, obs.max_clue_tokens, r.sig_play,
                obs.play_stacks, obs.colors)
            if not sig:
                continue
            score, ok = 0.0, True
            for o, act in sig:
                card = cardof[o]
                if act == "play":
                    if not obs.is_playable(card):
                        ok = False
                        break
                    score += self._play_value(obs, r, o, card, deducible, idmap)
                else:
                    score += self._discard_value(obs, r, o, card, deducible_trash)
            if not ok:
                continue
            action = (Action.clue_color(bob, color) if is_color
                      else Action.clue_rank(bob, rank))
            out.append((score, 1, action))
        return out

    def _reactive_candidates(self, obs, r, idmap, ct):
        me, n = obs.player_index, obs.num_players
        cathy = (me + 2) % n
        bob = (me + 1) % n
        E = self._expected_signal(obs, cathy, r)
        if E is None:
            return []
        chand = obs.hands[cathy]
        corders = [cv.order for cv in chand]
        e_card = next((cv.card for cv in chand if cv.order == E[2]), None)
        cathy_deducible = self._other_deducible_plays(obs, r, cathy)
        if self.GOODTOUCH_DEDUCIBLE_DUP and self.ASSUME_TEAMMATES_DEDUCE \
                and e_card is not None:
            if e_card in {cv.card for cv in chand if cv.order in cathy_deducible}:
                return []
        cathy_deducible_trash = self._other_deducible_trash(obs, r, cathy)
        if E[0] == "play":
            exp_score = self._play_value(obs, r, E[2], e_card, cathy_deducible, idmap)
        else:
            exp_score = self._discard_value(obs, r, E[2], e_card, cathy_deducible_trash)

        bhand = obs.hands[bob]
        bslots, bby = self._hand_slots([cv.order for cv in bhand])
        bcard = {cv.order: cv.card for cv in bhand}
        bob_handsize = len(bhand)
        bob_deducible = self._other_deducible_plays(obs, r, bob)
        bob_deducible_trash = self._other_deducible_trash(obs, r, bob)
        signalled_plays = [(cv.order, (cv.card.color, cv.card.rank))
                           for hand in (bhand, chand) for cv in hand
                           if cv.order in r.sig_play and cv.card is not None]
        in_zone = self._in_endgame(obs.deck_size, sum(obs.play_stacks.values()),
                                   obs.num_players, len(obs.colors) * 5)
        options = [(True, c, None) for c in {cv.card.color for cv in chand}]
        options += [(False, None, rk) for rk in {cv.card.rank for cv in chand}]
        out = []
        for is_color, color, rank in options:
            touched = [cv.order for cv in chand
                       if (cv.card.color == color if is_color else cv.card.rank == rank)]
            if (not self.STALLS_STABLE_ONLY
                    and self._stall_match(in_zone, is_color, color if is_color else rank,
                                          touched, r.rank_known, r.color_known)):
                continue
            if (self.EIGHT_CLUE_STALL and self.EIGHT_CLUE_STALL_REACTIVE
                    and (not is_color) and rank in self.EIGHT_CLUE_STALL_RANKS
                    and obs.clue_tokens == obs.max_clue_tokens):
                continue  # reads as a Cathy-directed 8-clue stall, not a reaction
            I = self._compute_initial(corders, touched, r.clued, r.color_known,
                                      r.rank_known, is_color)
            if I is None:
                continue
            flip = (I[0] != E[0])
            k = (I[1] - E[1]) % len(chand)
            bslot = 1 + k
            if bslot > bob_handsize:
                continue
            order = bby[bslot]
            card = bcard[order]
            if not flip and any(o != order and ident == (card.color, card.rank)
                                for o, ident in signalled_plays):
                continue
            intermediate = E[3]
            if intermediate is not None:
                if flip or card != intermediate:
                    continue
            elif not flip:
                if not obs.is_playable(card):
                    continue
                if e_card is not None and card == e_card:
                    continue
            else:
                if obs.clue_tokens - 1 >= obs.max_clue_tokens:
                    continue
                if not self._is_trash(obs, order, card, r):
                    continue
            if not flip:
                react_score = self._play_value(obs, r, order, card, bob_deducible, idmap)
            else:
                react_score = self._discard_value(obs, r, order, card, bob_deducible_trash)
            action = (Action.clue_color(cathy, color) if is_color
                      else Action.clue_rank(cathy, rank))
            out.append((exp_score + react_score, 0, action))
        return out

    # --- flowchart ------------------------------------------------------
    def act(self, obs: Observation) -> Action:
        in_zone = self._in_endgame(obs.deck_size, sum(obs.play_stacks.values()),
                                   obs.num_players, len(obs.colors) * 5)
        if in_zone:
            return self._endgame_act(obs)
        return self._scored_act(obs)

    def _scored_act(self, obs: Observation) -> Action:
        r = self._derive(obs)
        me, n = obs.player_index, obs.num_players
        # 1. forced reaction
        if r.pending is not None and n == 3:
            giver, _t, _I = r.pending
            if (giver + 1) % n == me:
                react = self._react(obs, r, r.pending)
                if react is not None:
                    return react
        # 2. play (deduced, then signalled)
        for play in (self._extra_play(obs, r), self._play_signaled(obs, r)):
            if play is not None:
                return play
        clues = self._clue_candidates(obs, r)
        best = min(clues, key=self._clue_key) if clues else None
        # 3. best clue scoring above the (clue-count-dependent) threshold
        if best is not None and best[0] > self._step3_threshold(obs.clue_tokens):
            return best[2]
        # 4. discard a discard-signal / deduced trash
        if obs.clue_tokens < obs.max_clue_tokens:
            d = self._discard_signaled(obs, r) or self._trash_discard(obs)
            if d is not None:
                return d
        # 5. best clue scoring positively
        if best is not None and best[0] > 0:
            return best[2]
        # 6. discard the chop (leftmost untouched)
        if obs.clue_tokens < obs.max_clue_tokens:
            d6 = self._discard_chop(obs, r)
            if d6 is not None:
                return d6
        # forced fallback
        if obs.clue_tokens > 0:
            st = self._stall_clue(obs)
            if st is not None:
                return st
        return self._fallback(obs, r)


class ReactorBridgePlayer(ReactorScoredPlayer):
    """Intermediate between rdend and rdscore: rdend's exact flowchart, but a
    reactive clue scoring <= 2 is deprioritised BELOW discarding a discard-signal
    / trash card (rdend gives every reactive clue above discarding). The reactive
    clue is still selected by rdend's `_reactive_clue`; only its score gates where
    it sits relative to the discard. No other rdscore changes (no skip-signalled,
    no stable/reactive score pooling)."""

    name = "rdbridge"
    EXPECTED_SKIP_SIGNALLED = False  # use rdend's expected signal, not the skip
    #: Reactive-clue selection: False = rdend's `_reactive_key` heuristic;
    #: True = the highest-scoring legal reactive clue.
    REACTIVE_MAX_SCORE = False
    #: Gate placement: False = only the SELECTED reactive clue, if it scores above
    #: the gate, goes above discarding; True = a reactive clue goes above discarding
    #: whenever ANY legal reactive clue clears the gate (selecting the best).
    REACTIVE_GATE_ANY = False

    #: Reactive-clue gate: a reactive clue must score strictly above this to be
    #: given above discarding. REACT_GATE_LOW (if set) overrides it at <=1 clue --
    #: e.g. 4.0 means no reactive clue can clear it there, so discarding a
    #: discard-signal/trash always wins when clues are scarce.
    REACT_GATE = 2.0
    REACT_GATE_LOW = None

    #: If set, when the chosen reactive clue scores <= this, prefer a stable play
    #: clue over it (a stable play clue otherwise sits below the low-value reactive).
    STABLE_OVER_REACTIVE_MAX = None

    def _react_gate(self, clue_tokens: int) -> float:
        if self.REACT_GATE_LOW is not None and clue_tokens <= 1:
            return self.REACT_GATE_LOW
        return self.REACT_GATE

    def act(self, obs: Observation) -> Action:
        in_zone = self._in_endgame(obs.deck_size, sum(obs.play_stacks.values()),
                                   obs.num_players, len(obs.colors) * 5)
        if in_zone:
            return self._endgame_act(obs)
        return self._bridge_act(obs)

    def _select(self, options):
        """Pick one (key, score, action) from reactive options."""
        if self.REACTIVE_MAX_SCORE:
            return min(options, key=lambda o: (-o[1], o[0]))
        return min(options, key=lambda o: o[0])

    def _reactive_options(self, obs, r):
        """Every valid reactive clue as (reactive_key, score, action). Mirrors
        rdend's `_reactive_clue` enumeration (so `_reactive_key` selection matches
        it exactly) while also attaching the rdscore additive score."""
        me, n = obs.player_index, obs.num_players
        if obs.clue_tokens <= 0 or n != 3:
            return []
        cathy, bob = (me + 2) % n, (me + 1) % n
        E = self._expected_signal(obs, cathy, r)
        if E is None:
            return []
        chand = obs.hands[cathy]
        corders = [cv.order for cv in chand]
        e_card = next((cv.card for cv in chand if cv.order == E[2]), None)
        idmap = {cv.order: (cv.card.color, cv.card.rank)
                 for p in range(n) if p != me
                 for cv in obs.hands[p] if cv.card is not None}
        cathy_ded = self._other_deducible_plays(obs, r, cathy)
        if E[0] == "play":
            exp = self._play_value(obs, r, E[2], e_card, cathy_ded, idmap)
        else:
            exp = self._discard_value(obs, r, E[2], e_card,
                                      self._other_deducible_trash(obs, r, cathy))
        bhand = obs.hands[bob]
        bslots, bby = self._hand_slots([cv.order for cv in bhand])
        bcard = {cv.order: cv.card for cv in bhand}
        bob_handsize = len(bhand)
        leftmost_nopsig = min((bslots[cv.order] for cv in bhand
                               if cv.order not in r.sig_play), default=99)
        signalled_plays = [(cv.order, (cv.card.color, cv.card.rank))
                           for hand in (bhand, chand) for cv in hand
                           if cv.order in r.sig_play and cv.card is not None]
        bob_ded = (self._other_deducible_plays(obs, r, bob)
                   if self.ASSUME_TEAMMATES_DEDUCE else set())
        bob_ded_trash = self._other_deducible_trash(obs, r, bob)
        in_zone = self._in_endgame(obs.deck_size, sum(obs.play_stacks.values()),
                                   obs.num_players, len(obs.colors) * 5)
        opts = [(True, c, None) for c in {cv.card.color for cv in chand}]
        opts += [(False, None, rk) for rk in {cv.card.rank for cv in chand}]
        out = []
        for is_color, color, rank in opts:
            touched = [cv.order for cv in chand
                       if (cv.card.color == color if is_color else cv.card.rank == rank)]
            if (not self.STALLS_STABLE_ONLY
                    and self._stall_match(in_zone, is_color, color if is_color else rank,
                                          touched, r.rank_known, r.color_known)):
                continue
            if (self.EIGHT_CLUE_STALL and self.EIGHT_CLUE_STALL_REACTIVE
                    and (not is_color) and rank in self.EIGHT_CLUE_STALL_RANKS
                    and obs.clue_tokens == obs.max_clue_tokens):
                continue  # reads as a Cathy-directed 8-clue stall, not a reaction
            I = self._compute_initial(corders, touched, r.clued, r.color_known,
                                      r.rank_known, is_color)
            if I is None:
                continue
            flip = (I[0] != E[0])
            k = (I[1] - E[1]) % len(chand)
            bslot = 1 + k
            if bslot > bob_handsize:
                continue
            order = bby[bslot]
            card = bcard[order]
            if not flip and any(o != order and ident == (card.color, card.rank)
                                for o, ident in signalled_plays):
                continue
            intermediate = E[3]
            if intermediate is not None:
                if flip or card != intermediate:
                    continue
            elif not flip:
                if not obs.is_playable(card):
                    continue
                if e_card is not None and card == e_card:
                    continue
            else:
                if obs.clue_tokens - 1 >= obs.max_clue_tokens:
                    continue
                if not self._is_trash(obs, order, card, r):
                    continue
            if not flip:
                rscore = self._play_value(obs, r, order, card, bob_ded, idmap)
            else:
                rscore = self._discard_value(obs, r, order, card, bob_ded_trash)
            dprio = self._discard_priority(obs, card) if flip else 0
            redundant = (not flip) and (order in bob_ded)
            key = self._reaction_key(obs, r, flip, bslot, k, leftmost_nopsig,
                                     dprio, redundant, order, card, idmap,
                                     bob_ded, cathy_ded)
            act = (Action.clue_color(cathy, color) if is_color
                   else Action.clue_rank(cathy, rank))
            out.append((key, exp + rscore, act))
        return out

    def _reaction_key(self, obs, r, flip, bslot, k, leftmost_nopsig, dprio,
                      redundant, order, card, idmap, bob_ded, cathy_ded):
        """Sort key for a reaction (lower = preferred). Default = rdend's
        `_reactive_key`; subclasses may insert extra criteria."""
        return self._reactive_key(obs, flip, bslot, k, leftmost_nopsig, dprio,
                                  redundant)

    def _five_save_action(self, obs: Observation, r) -> Action | None:
        """When Bob's chop is a critical card he'd blind-discard and Alice has no
        reaction to make, give a clue over playing/discarding. Prefer any normal
        clue (it makes Bob act on it instead of chopping); with FIVE_SAVE_ON_CHOP
        it falls back to a dedicated stable rank-5 '5-save' when the chop is a 5.
        Returns None when the trigger doesn't hold. Scope: CLUE_OVER_CRIT_CHOP =>
        any critical chop; otherwise 5s only."""
        if not (self.FIVE_SAVE_ON_CHOP or self.CLUE_OVER_CHOP5
                or self.CLUE_OVER_CRIT_CHOP or self.CLUE_OVER_PLAYABLE_CHOP
                or self.CLUE_OVER_TWO_CHOP) \
                or obs.clue_tokens < self.CHOP_CLUE_MIN:
            return None
        me, n = obs.player_index, obs.num_players
        if n != 3:
            return None
        bob, cathy = (me + 1) % n, (me + 2) % n
        # 3. Alice must not need to react.
        if r.pending is not None and (r.pending[0] + 1) % n == me:
            return None
        # Bob's chop must be worth saving (any critical card, or a 5).
        chop = self._chop_order(obs.hands[bob], r)
        if chop is None:
            return None
        bcard = {cv.order: cv.card for cv in obs.hands[bob]}
        chop_card = bcard.get(chop)
        if chop_card is None:
            return None
        if self.CLUE_OVER_CRIT_CHOP:
            worth = self._is_critical(obs, chop_card)
        else:  # FIVE_SAVE_ON_CHOP / CLUE_OVER_CHOP5: 5s only
            worth = chop_card.rank == 5
        # ...also a playable chop with no visible duplicate (a wasted ready point).
        if not worth and self.CLUE_OVER_PLAYABLE_CHOP:
            worth = (obs.is_playable(chop_card)
                     and not self._card_seen_in_other_hand(obs, chop, chop_card))
        # ...also a rank-2 whose other copy Alice can't see (early suit insurance).
        if not worth and self.CLUE_OVER_TWO_CHOP:
            worth = (chop_card.rank == 2
                     and not self._card_seen_in_other_hand(obs, chop, chop_card))
        if not worth:
            return None
        # 1. Bob has no play / deducible play / discard / deducible discard.
        bset = {cv.order for cv in obs.hands[bob]}
        if any(o in r.sig_play for o in bset) or self._other_deducible_plays(obs, r, bob):
            return None
        if any(o in r.sig_disc for o in bset) or self._other_deducible_trash(obs, r, bob):
            return None
        # 2. Bob has no stable clue to give to Cathy (else he'd clue over chopping).
        if (self._stable_clue(obs, r, True, giver=bob, target=cathy) is not None
                or self._stable_clue(obs, r, False, giver=bob, target=cathy) is not None):
            return None
        # Trigger holds: any clue over play/discard, 5-save as the last resort.
        options = self._reactive_options(obs, r)
        chosen = self._select(options) if options else None
        if chosen is not None:
            return chosen[2]
        sp = self._stable_clue(obs, r, want_play=True)
        if sp is not None:
            return sp
        sd = self._stable_clue(obs, r, want_play=False)
        if sd is not None:
            return sd
        # dedicated rank-5 save only when the full convention is on and it's a 5
        if self.FIVE_SAVE_ON_CHOP and chop_card.rank == 5:
            return Action.clue_rank(bob, 5)
        return None

    def _bridge_act(self, obs: Observation) -> Action:
        r = self._derive(obs)
        me, n = obs.player_index, obs.num_players
        # 1. forced reaction
        if r.pending is not None and n == 3:
            giver, _t, _I = r.pending
            if (giver + 1) % n == me:
                react = self._react(obs, r, r.pending)
                if react is not None:
                    return react
        # 1.5 clue over play/discard when Bob's critical/playable chop is in danger
        if (self.FIVE_SAVE_ON_CHOP or self.CLUE_OVER_CHOP5
                or self.CLUE_OVER_CRIT_CHOP or self.CLUE_OVER_PLAYABLE_CHOP
                or self.CLUE_OVER_TWO_CHOP):
            fs = self._five_save_action(obs, r)
            if fs is not None:
                return fs
        # 2. play (deduced, then signalled)
        for play in (self._extra_play(obs, r), self._play_signaled(obs, r)):
            if play is not None:
                return play
        # 3. reactive clue above discarding, gated by score > threshold.
        gate = self._react_gate(obs.clue_tokens)
        options = self._reactive_options(obs, r)
        chosen = self._select(options) if options else None
        if self.REACTIVE_GATE_ANY:
            above = [o for o in options if o[1] > gate]
            gate_pick = self._select(above) if above else None
        else:
            gate_pick = chosen if (chosen is not None and chosen[1] > gate) else None
        if gate_pick is not None:
            return gate_pick[2]
        # discard a discard-signal / trash card, above a low-value reactive clue
        if obs.clue_tokens < obs.max_clue_tokens:
            d = self._discard_signaled(obs, r) or self._trash_discard(obs)
            if d is not None:
                return d
        # if the reactive clue is weak (<= STABLE_OVER_REACTIVE_MAX), prefer a
        # stable play clue over it.
        if (self.STABLE_OVER_REACTIVE_MAX is not None and chosen is not None
                and chosen[1] <= self.STABLE_OVER_REACTIVE_MAX and obs.clue_tokens > 0):
            sp = self._stable_clue(obs, r, want_play=True)
            if sp is not None:
                return sp
        # the low-value reactive clue now sits here
        if chosen is not None:
            return chosen[2]
        # rest of rdend's flowchart unchanged
        if obs.clue_tokens > 0:
            sp = self._stable_clue(obs, r, want_play=True)
            if sp is not None:
                return sp
        if obs.clue_tokens > 0:
            sd = self._stable_clue(obs, r, want_play=False)
            if sd is not None:
                return sd
        if obs.clue_tokens < obs.max_clue_tokens:
            ch = self._discard_chop(obs, r)
            if ch is not None:
                return ch
        if obs.clue_tokens > 0:
            st = self._stall_clue(obs)
            if st is not None:
                return st
        return self._fallback(obs, r)


class ReactorBridge4Player(ReactorBridgePlayer):
    """rdbridge, but the play-reaction tiebreak chain is:
      1.  non-redundant (Bob can't already deduce it playable)
      2.  prefer a card with no play signal
      2.1 prefer a card with no other copy already queued-to-play / deducibly playable
      2.2 prefer a card not yet touched (clued)
      2.3 prefer a card with no other copy already touched
      3.  leftmost (lower slot), then k
    (criterion 2 is now the broad 'no play signal', so 2.1-2.3 actually break ties
    before the leftmost-slot preference.)"""

    name = "rdbridge4"

    def _reaction_key(self, obs, r, flip, bslot, k, leftmost_nopsig, dprio,
                      redundant, order, card, idmap, bob_ded, cathy_ded):
        base = self._reactive_key(obs, flip, bslot, k, leftmost_nopsig, dprio,
                                  redundant)
        if flip:
            return base  # discard reaction: unchanged
        pid = (card.color, card.rank)
        accounted = r.sig_play | bob_ded | cathy_ded
        dup_queued = any(o2 != order and idmap.get(o2) == pid for o2 in accounted)
        touched = order in r.clued
        dup_touched = any(o2 != order and idmap.get(o2) == pid for o2 in r.clued)
        wrapper, play_key = base
        red, _slot_match, bslot_, k_ = play_key
        new_play_key = (red,
                        1 if order in r.sig_play else 0,  # 2. prefer no play signal
                        1 if dup_queued else 0,            # 2.1
                        1 if touched else 0,               # 2.2
                        1 if dup_touched else 0,           # 2.3
                        bslot_, k_)                        # 3. leftmost, then k
        return (wrapper, new_play_key)


class ReactorPtrNoSkipPlayer(ReactorBridge4Player):
    """rdbridge4 but the rank-clue discard pointer skips only PREVIOUSLY-clued
    cards, not cards clued by the same clue (so a just-clued card can be the
    discard target)."""

    name = "rdptrnoskip"
    DISCARD_PTR_SKIP_NEW = False


class ReactorCritPlayChopPlayer(ReactorPtrNoSkipPlayer):
    """CURRENT BEST (~80.75%). rdptrnoskip + "clue over play/discard when the next
    player would blind-discard a card worth saving": Alice gives a normal clue
    (over her own play/discard) whenever Bob would chop a CRITICAL card (last copy
    of a still-needed card) or a currently-PLAYABLE card with no visible duplicate
    -- the clue makes him act on it instead of chopping. Plus today's deduction /
    expected-signal fixes on the base (DEDUCE_OWN_KNOWN, EXPECTED_DEDUP_SIGNAL).

    Lineage / A/B (20k, vs rdptrnoskip 78.82%): forcing a clue on a doomed chop-5
    only (CLUE_OVER_CHOP5, "rd5force") = 79.99% (+1.18pp, the bulk); generalize to
    any critical chop (CLUE_OVER_CRIT_CHOP, "rdcritclue") = 80.08%; + playable-no-dup
    (CLUE_OVER_PLAYABLE_CHOP, this) = 80.34% and the best mean; + own-known deduction
    = 80.41%; + clean expected signal = 80.75%. All 0% strikeout.

    Pruned/parked (flags kept on the base w/ notes): the full rank-5 "5-save"
    CONVENTION (FIVE_SAVE_ON_CHOP, "rd5save" 80.47%) -- dropped as too much mental
    effort for human partners; BAN_STABLE_5_ON_CHOP (-0.34pp, measurement only);
    CHOP_CLUE_MIN>1 ("rdcritplay2", neutral); CLUE_OVER_TWO_CHOP ("rdcritplaytwo",
    neutral-to-negative)."""

    name = "rdcritplay"
    CLUE_OVER_CRIT_CHOP = True
    CLUE_OVER_PLAYABLE_CHOP = True
