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


class ReactorDeducePlayPriorityPlayer(ReactorDeducePlayer):
    """Experiment: a stable play-clue outranks discarding a discard-signalled
    card while >=2 clue tokens remain."""

    name = "rdplay"
    STABLE_PLAY_BEFORE_DISCARD_AT_2 = True


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
        best = None  # (tier, action)
        for rank in self.STALL_RANKS:
            for p in obs.other_players():
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


class ReactorGoodTouchPlayer(ReactorEndgamePlayer):
    """Experiment (NOT adopted): rdend + giver-side good-touch against deducible
    duplicates (GOODTOUCH_DEDUCIBLE_DUP). A/B (20k) 77.26% -> 76.94% perfect --
    the duplicate strikes it prevents are mostly harmless, and refusing those
    clues loses more tempo than it saves. Kept registered for reproducibility."""

    name = "rdgt"
    GOODTOUCH_DEDUCIBLE_DUP = True


class ReactorDeferDupPlayer(ReactorEndgamePlayer):
    """Experiment: rdend + DEDUCE_DEFER_DUP_SIGNAL -- defer a deduced play when an
    own play-signalled card might be its duplicate, playing the signalled (unknown)
    card first so the deduced (known) card stays re-checkable afterwards."""

    name = "rddefer"
    DEDUCE_DEFER_DUP_SIGNAL = True
