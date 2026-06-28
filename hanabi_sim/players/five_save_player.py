"""Play-clue convention extended with **5 saves** and a **5-clue stall**.

Fives are special: there's only one of each, so a discarded 5 caps that color at
4. The play-clue bot discards chop blindly and loses 5s. This strategy adds a
dedicated channel for 5s, keeping it cleanly separate from play clues so nothing
is ever misread.

Conventions added on top of :class:`PlayCluePlayer`:

* **A 5 save is not a play call.** A rank-5 clue is judged against the stacks at
  the time it was given (see ``PlayCluePlayer._derive_called``); with no stack at
  4 it isn't consistent with being playable, so it is read as "save", not "play".
  A saved 5 is only ever played once it is fully known and its stack is ready
  (via ``known_playable``). No special-casing of 5s is needed for this.
* **Play-clue a 5 only by COLOR**, and never one the receiver already knows is
  playable. A rank-5 clue means "save", so it can't double as a play clue; a
  color clue leaves the rank open, so the 5 is read as a play call and played.
  (Other 5s still resolve via the save / color-fill / auto-play flow.)
* **5 save:** if the next player has no play and a 5 is sitting on their chop
  (and they don't already know it's a 5), clue rank 5 to save it.
* **Skip 5s when discarding:** a card known to be a 5 is never chopped.
* **5-clue stalls** (when forced to clue with nothing better), in order:
  1. a color clue that fills in the color of a card already known to be a 5,
  2. a rank-5 clue that newly informs a 5,
  3. a rank-5 clue to cards already known to be 5s (pure stall),
  4. the base stall (a >=2-card clue that can't be misread, else any legal clue).
"""

from __future__ import annotations

from ..actions import Action
from ..observation import CardView, Observation
from .play_clue_player import PlayCluePlayer


class FiveSavePlayer(PlayCluePlayer):
    name = "fivesave"

    def act(self, obs: Observation) -> Action:
        # 1) Urgent: stop the next player from discarding a 5 off their chop.
        if obs.clue_tokens > 0:
            save = self._five_save(obs)
            if save is not None:
                return save

        # 2) Play a card we know -- or have been told -- is playable.
        play = self._choose_play(obs)
        if play is not None:
            return play

        # 3) Signal a teammate's playable (1-4) card with a play clue.
        if obs.clue_tokens > 0:
            clue = self._choose_play_clue(obs)
            if clue is not None:
                return clue

        # 4) Discard (chop), skipping known 5s.
        if obs.clue_tokens < obs.max_clue_tokens:
            return self._choose_discard(obs)

        # 5) Forced: stall, preferring 5-clues.
        return self._stall_clue(obs)

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def _is_known_five(view: CardView) -> bool:
        return view.possible_ranks == {5}

    def _derive_called(self, obs: Observation, target: int) -> set[int]:
        """As the base, but a card *known to be a 5* is never a play call.

        A rank-5 clue is a save/inform here, never "play it". Clue-time freezing
        alone isn't enough: a 5 saved while *another* color's stack is already at
        4 would be consistent with being that playable 5 at clue time. Excluding
        known 5s closes that gap -- a 5 is only ever played via ``known_playable``
        once its color is known and its stack is ready.
        """
        called = super()._derive_called(obs, target)
        views = {cv.order: cv for cv in obs.hands[target]}
        return {
            o for o in called
            if not (o in views and self._is_known_five(views[o]))
        }

    def _player_has_play(self, obs: Observation, p: int) -> bool:
        """Would player ``p`` have something to play on their turn?"""
        called_p = self._derive_called(obs, p)
        for cv in obs.hands[p]:
            if self._view_known_playable(cv, obs.play_stacks) or cv.order in called_p:
                return True
        return False

    def _chop_index(self, obs: Observation, p: int) -> int | None:
        """The card player ``p`` would discard: their oldest unclued card."""
        for i, cv in enumerate(obs.hands[p]):
            if not cv.clued:
                return i
        return None

    # --- 1) five save -----------------------------------------------------
    def _five_save(self, obs: Observation) -> Action | None:
        nxt = (obs.player_index + 1) % obs.num_players
        if nxt == obs.player_index:
            return None
        if self._player_has_play(obs, nxt):
            return None  # they won't discard this turn, so the 5 is safe for now
        chop = self._chop_index(obs, nxt)
        if chop is None:
            return None
        cv = obs.hands[nxt][chop]
        if cv.card.rank == 5 and not self._is_known_five(cv):
            return Action.clue_rank(nxt, 5)
        return None

    # --- 2) play: inherited. A saved 5 isn't a play call (frozen at clue
    #        time), and a fully-known, ready 5 plays via known_playable. ----

    # --- 3) play clue (override: play-clue 5s only by COLOR) --------------
    def _play_clue_for(self, obs, p, cv, colors, ranks):
        card = cv.card
        if card.rank == 5:
            # Rule: never play-clue a 5 the receiver already knows is playable.
            if self._view_known_playable(cv, obs.play_stacks):
                return None
            # Use COLOR only. A rank-5 clue reads as a save (and a resulting
            # known-5 is excluded from play calls), so it would never be played;
            # a color clue leaves the rank open, so the 5 is called and plays.
            if colors.count(card.color) == 1:
                return Action.clue_color(p, card.color)
            return None
        return super()._play_clue_for(obs, p, cv, colors, ranks)

    # --- 4) discard (override: skip known 5s) -----------------------------
    def _choose_discard(self, obs: Observation) -> Action:
        called = self._called_orders(obs, target=obs.player_index)
        hand = obs.own_hand

        # Known-dead cards are always safe to dump.
        for i in range(len(hand)):
            if obs.known_dead(i):
                return Action.discard(i)
        # Chop: oldest card that is unclued, uncalled, and not a known 5.
        for i, cv in enumerate(hand):
            if not cv.clued and cv.order not in called and not self._is_known_five(cv):
                return Action.discard(i)
        # Forced down: oldest unclued/uncalled card (may be a 5).
        for i, cv in enumerate(hand):
            if not cv.clued and cv.order not in called:
                return Action.discard(i)
        # Everything is protected; avoid known 5s if we still can.
        for i, cv in enumerate(hand):
            if cv.order not in called and not self._is_known_five(cv):
                return Action.discard(i)
        for i, cv in enumerate(hand):
            if cv.order not in called:
                return Action.discard(i)
        return Action.discard(0)

    # --- 5) stall (override: prefer 5-clues) ------------------------------
    def _stall_clue(self, obs: Observation) -> Action:
        return (
            self._color_fill_five_clue(obs)
            or self._informative_five_clue(obs)
            or self._redundant_five_clue(obs)
            or super()._stall_clue(obs)
        )

    def _color_fill_five_clue(self, obs: Observation) -> Action | None:
        """Color clue revealing the color of a card already known to be a 5.

        Safe to single-touch: the holder already knows it's a 5, so it can't be
        misread as a play clue. Filling the color lets them play it later.
        """
        for p in obs.other_players():
            for cv in obs.hands[p]:
                if self._is_known_five(cv) and len(cv.possible_colors) > 1:
                    return Action.clue_color(p, cv.card.color)
        return None

    def _informative_five_clue(self, obs: Observation) -> Action | None:
        """Rank-5 clue that tells a teammate about a 5 they don't yet know."""
        for p in obs.other_players():
            if any(cv.card.rank == 5 and not self._is_known_five(cv) for cv in obs.hands[p]):
                return Action.clue_rank(p, 5)
        return None

    def _redundant_five_clue(self, obs: Observation) -> Action | None:
        """Rank-5 clue even if all touched 5s are already known (pure stall)."""
        for p in obs.other_players():
            if any(cv.card.rank == 5 for cv in obs.hands[p]):
                return Action.clue_rank(p, 5)
        return None
