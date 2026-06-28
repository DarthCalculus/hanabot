"""A bot built around the *play-clue* convention (tuned for 3 players).

The convention
--------------
A clue that touches **exactly one** card is a "play clue". The receiver trusts
that a teammate only does this for a card that is currently playable, so they
play that card -- provided the clue information is still consistent with it being
playable.

This bot implements both sides:

* **Receiving** -- if one of my own cards was the sole card touched by a clue
  aimed at me, and some value it could still be is playable right now, I play it.
* **Giving** -- I look for a teammate's currently-playable card and clue it in a
  way that touches only that card. I only ever single-clue genuinely playable
  cards, and I never signal two copies of the same card, so every singleton clue
  a teammate receives is a trustworthy play clue.

What it does *not* do yet: save clues, finesses, prompts, or chop management. It
will therefore sometimes discard a card the team still needs, which caps the
score it can reach. That's the motivation for the next convention to add.
"""

from __future__ import annotations

from ..actions import Action, ActionRecord, ActionType
from ..observation import CardView, Observation
from .base import Player


class PlayCluePlayer(Player):
    name = "playclue"

    def act(self, obs: Observation) -> Action:
        # 1) Play a card we know -- or have been told -- is playable.
        play = self._choose_play(obs)
        if play is not None:
            return play

        # 2) Signal a teammate's playable card with a single-card play clue.
        if obs.clue_tokens > 0:
            clue = self._choose_play_clue(obs)
            if clue is not None:
                return clue

        # 3) Otherwise discard (chop), if discarding is legal.
        if obs.clue_tokens < obs.max_clue_tokens:
            return self._choose_discard(obs)

        # 4) Forced: at max tokens with nothing better, give a safe stall clue.
        return self._stall_clue(obs)

    # --- shared inference -------------------------------------------------
    @staticmethod
    def _called_orders(obs: Observation, target: int | None = None) -> set[int]:
        """Orders of cards that were the sole card touched by some clue.

        Under this convention a single-card clue *is* a play clue, so these are
        exactly the cards that have been "called to play". Pass ``target`` to
        restrict to clues aimed at one player.
        """
        called: set[int] = set()
        for rec in obs.log:
            if rec.action.is_clue and len(rec.touched_orders) == 1:
                if target is None or rec.action.target == target:
                    called.add(rec.touched_orders[0])
        return called

    @staticmethod
    def _consistent_playable(view: CardView, stacks) -> bool:
        """True if some value still possible for ``view`` plays against ``stacks``."""
        for color in view.possible_colors:
            if stacks[color] + 1 in view.possible_ranks:
                return True
        return False

    def _derive_called(self, obs: Observation, target: int) -> set[int]:
        """Orders of ``target``'s cards that have been *called to play*.

        A single-card clue is a play clue, judged against the stacks **as they
        were when the clue was given** -- reconstructed by replaying the log's
        successful plays -- and never re-judged against later stacks. (This is
        why a 5 saved while no stack is at 4 is not a play call, even if some
        stack reaches 4 afterwards.) A card whose *current* knowledge makes it
        unplayable forever is dropped, so we never knowingly misplay.
        """
        stacks = {c: 0 for c in obs.colors}
        views = {cv.order: cv for cv in obs.hands[target]}
        called: set[int] = set()
        for rec in obs.log:
            a = rec.action
            if a.is_clue and a.target == target:
                for order in self._clue_play_targets(rec, stacks):
                    cv = views.get(order)  # still in hand?
                    # Judge against clue-time stacks, using best current knowledge.
                    if cv is not None and self._consistent_playable(cv, stacks):
                        called.add(order)
            elif a.type is ActionType.PLAY and rec.success:
                stacks[rec.played_card.color] = rec.played_card.rank
        return called

    def _derive_trash(self, obs: Observation, target: int) -> set[int]:
        """Orders of ``target``'s cards a clue signalled as *discard this*.

        None by default; conventions that add discard signals (e.g. trash-1
        clues) override ``_clue_discard_targets``.
        """
        stacks = {c: 0 for c in obs.colors}
        views = {cv.order: cv for cv in obs.hands[target]}
        trash: set[int] = set()
        for rec in obs.log:
            a = rec.action
            if a.is_clue and a.target == target:
                for order in self._clue_discard_targets(rec, stacks):
                    if order in views:
                        trash.add(order)
            elif a.type is ActionType.PLAY and rec.success:
                stacks[rec.played_card.color] = rec.played_card.rank
        return trash

    def _clue_play_targets(self, rec, stacks) -> tuple:
        """Which of a clue's touched cards it calls as plays (before the
        playable filter). Default convention: a single-card clue calls its one
        focused card; multi-card clues call nothing. ``stacks`` are the stacks
        as of the clue (for count-based conventions).
        """
        return rec.touched_orders if len(rec.touched_orders) == 1 else ()

    def _clue_discard_targets(self, rec, stacks) -> tuple:
        """Which touched cards a clue signals to discard. None by default."""
        return ()

    def _clue_action_targets(self, rec, stacks) -> tuple:
        """All cards the receiver would act on (play or discard) from a clue."""
        return tuple(self._clue_play_targets(rec, stacks)) + tuple(
            self._clue_discard_targets(rec, stacks)
        )

    @staticmethod
    def _view_known_playable(view: CardView, stacks) -> bool:
        """True if every value still possible for ``view`` plays right now (so its
        holder will play it on their own)."""
        if not view.possible_colors or not view.possible_ranks:
            return False
        return all(
            r == stacks[c] + 1 for c in view.possible_colors for r in view.possible_ranks
        )

    def _will_play(self, obs: Observation) -> tuple[set, set]:
        """(orders, identities) of cards some player will play -- whether called
        by a clue or already known-playable to its holder. A play clue must never
        target one of these, nor a duplicate of one (good touch)."""
        orders: set[int] = set()
        ids: set = set()
        for p in range(obs.num_players):
            called = self._derive_called(obs, p)
            for cv in obs.hands[p]:
                if cv.order in called or self._view_known_playable(cv, obs.play_stacks):
                    orders.add(cv.order)
                    if cv.card is not None:  # identity visible (not our own hand)
                        ids.add((cv.card.color, cv.card.rank))
        return orders, ids

    # --- 1) play ----------------------------------------------------------
    def _choose_play(self, obs: Observation) -> Action | None:
        called = self._derive_called(obs, obs.player_index)
        best_i: int | None = None
        best_rank = 99
        for i, view in enumerate(obs.own_hand):
            certain = obs.known_playable(i)
            # A frozen play call -- but only act on it if some value the card
            # could still be is actually playable *now*. If nothing legal fits
            # the clue against the current stacks, the "play clue" doesn't apply
            # (treat it as a stall) rather than bombing a dead/future card.
            told = view.order in called and self._consistent_playable(view, obs.play_stacks)
            if certain or told:
                # Prefer the lowest-ranked candidate (build foundations first).
                rank_guess = min(view.possible_ranks) if view.possible_ranks else 99
                if rank_guess < best_rank:
                    best_rank = rank_guess
                    best_i = i
        return None if best_i is None else Action.play(best_i)

    # --- 2) give a play clue ---------------------------------------------
    def _choose_play_clue(self, obs: Observation) -> Action | None:
        # Never play-clue a card that will already be played -- by a clue call or
        # by its holder's own certainty -- nor a duplicate of one (good touch).
        will_orders, will_ids = self._will_play(obs)

        candidates: list[tuple[tuple, Action]] = []  # (sort key, clue)
        for p in obs.other_players():
            hand = obs.hands[p]
            colors = [cv.card.color for cv in hand]
            ranks = [cv.card.rank for cv in hand]
            for idx, cv in enumerate(hand):
                card = cv.card
                if not obs.is_playable(card):
                    continue
                if cv.order in will_orders:
                    continue  # already going to be played
                if (card.color, card.rank) in will_ids:
                    continue  # a copy of it is already going to be played
                clue = self._play_clue_for(obs, p, cv, colors, ranks)
                if clue is None:
                    continue  # no clean single-card clue (or subclass veto)
                candidates.append((self._play_clue_priority(obs, p, idx, cv), clue))

        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    def _clue_target_key(self, obs: Observation, p: int) -> tuple:
        """Priority over *which player* to clue (lower goes first). This sorts
        above the focus-rank tie-break, so subclasses choosing a player (e.g. the
        next one with nothing to play) take precedence over which card. Default:
        the soonest player to act.
        """
        return ((p - obs.current_player) % obs.num_players,)

    def _play_clue_priority(self, obs: Observation, p: int, idx: int, cv: CardView) -> tuple:
        """Sort key for single-card play-clue candidates: player choice first,
        then the lowest focus rank."""
        return self._clue_target_key(obs, p) + (cv.card.rank,)

    def _play_clue_for(
        self, obs: Observation, p: int, cv: CardView,
        colors: list, ranks: list,
    ) -> Action | None:
        """Which clue to use to play-clue ``cv`` (touching only it), or None.

        Subclasses override to restrict clue type per card (e.g. only color-clue
        5s). ``colors``/``ranks`` are player ``p``'s card colors/ranks, so a
        ``count == 1`` means the clue touches just this card.
        """
        card = cv.card
        if ranks.count(card.rank) == 1:
            return Action.clue_rank(p, card.rank)
        if colors.count(card.color) == 1:
            return Action.clue_color(p, card.color)
        return None

    # --- 3) discard -------------------------------------------------------
    def _choose_discard(self, obs: Observation) -> Action:
        called = self._called_orders(obs, target=obs.player_index)
        hand = obs.own_hand
        # Known-dead cards are always safe to dump.
        for i in range(len(hand)):
            if obs.known_dead(i):
                return Action.discard(i)
        # Chop: oldest card that is neither clued nor called to play.
        for i in range(len(hand)):
            if not hand[i].clued and hand[i].order not in called:
                return Action.discard(i)
        # Everything is protected; discard the oldest non-called card.
        for i in range(len(hand)):
            if hand[i].order not in called:
                return Action.discard(i)
        return Action.discard(0)

    # --- 4) forced stall --------------------------------------------------
    def _stall_clue(self, obs: Observation) -> Action:
        """A clue to burn when forced (max tokens, nothing better).

        Prefer a *harmless* clue: one that would make the receiver play only
        cards that are actually playable and discard only cards that are
        actually dead. (The giver can check, since it sees the cards.) This way
        a stall can never be misread into a misplay or a wasteful discard.
        Which cards a clue makes the receiver act on is decided by the
        convention hooks, so this stays correct as conventions are added. Falls
        back to any legal clue only if no harmless one exists.
        """
        safe: Action | None = None
        fallback: Action | None = None
        for p in obs.other_players():
            hand = obs.hands[p]
            cards = {cv.order: cv.card for cv in hand}
            options = [
                (Action.clue_color(p, c), [cv for cv in hand if cv.card.color == c])
                for c in {cv.card.color for cv in hand}
            ] + [
                (Action.clue_rank(p, r), [cv for cv in hand if cv.card.rank == r])
                for r in {cv.card.rank for cv in hand}
            ]
            for act, touched in options:
                fallback = fallback or act
                if safe is not None:
                    continue
                rec = ActionRecord(
                    turn=0, player=p, action=act,
                    touched_orders=tuple(cv.order for cv in touched),
                )
                plays = self._clue_play_targets(rec, obs.play_stacks)
                discards = self._clue_discard_targets(rec, obs.play_stacks)
                harmful = any(not obs.is_playable(cards[o]) for o in plays) or any(
                    cards[o].rank > obs.play_stacks[cards[o].color] for o in discards
                )
                if not harmful:
                    safe = act
        return safe or fallback
