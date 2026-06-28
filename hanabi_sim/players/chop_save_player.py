"""Wait-and-play critical save via a singleton number clue, on top of critsave.

A singleton rank-(2,3,4) clue is read as a *critical save* -- not a play -- only
when BOTH:
  (A) it touches the receiver's chop card (the oldest unclued card), and
  (B) a card of that rank is already in the discard pile.
(B) is what tells the receiver a critical (last surviving) copy of that rank can
even exist; without it, a singleton number clue is just a normal play clue. So
this leaves singleton number clues fully usable as plays everywhere else.

When it is a save, the receiver holds the card and plays it once **every** stack
is at r-1 or higher: the card is the last copy of a needed r, so its own stack
can't have advanced past r-1, and once all stacks reach r-1 its stack is exactly
r-1 -- guaranteed playable without knowing the color. This lets us save a
critical 2/3/4 even when its rank is currently playable, which plain critsave
couldn't.

The receiver needs the giver to never *accidentally* produce that signal, so when
giving a play clue or a stall we avoid a singleton number clue that would land on
a teammate's chop while that rank sits in the discard.
"""

from __future__ import annotations

from ..actions import Action, ActionType
from ..observation import Observation
from .critical_save_player import CriticalSavePlayer
from .focus_player import FocusPlayer


class ChopSavePlayer(CriticalSavePlayer):
    name = "chopsave"

    @staticmethod
    def _hand_size(obs: Observation) -> int:
        return 5 if obs.num_players <= 3 else 4

    @staticmethod
    def _rank_in_discard(obs: Observation, rank: int) -> bool:
        return any(c.rank == rank for c in obs.discard_pile)

    def _is_chop(self, obs: Observation, p: int, order: int) -> bool:
        chop = self._chop_index(obs, p)
        return chop is not None and obs.hands[p][chop].order == order

    # --- receiving: detect the wait-and-play saves ------------------------
    def _derive_saved_low(self, obs: Observation, target: int | None = None) -> dict:
        """{order: rank} of cards saved by a singleton number clue that, at the
        time it was given, touched ``target``'s chop while that rank was already
        in the discard. Reconstructs the hand/chop and discard timeline."""
        if target is None:
            target = obs.player_index
        n = obs.num_players
        hand = [target + n * k for k in range(self._hand_size(obs))]  # oldest first
        clued: set[int] = set()
        discard_ranks: set[int] = set()
        own_now = {cv.order for cv in obs.hands[target]}
        saved: dict = {}
        for rec in obs.log:
            a = rec.action
            if a.is_clue:
                if a.target == target:
                    if (a.type is ActionType.CLUE_RANK and a.rank in (2, 3, 4)
                            and len(rec.touched_orders) == 1):
                        chop = next((o for o in hand if o not in clued), None)
                        t = rec.touched_orders[0]
                        if t == chop and a.rank in discard_ranks and t in own_now:
                            saved[t] = a.rank
                    clued.update(rec.touched_orders)
            else:  # play / discard
                if rec.player == target and a.card_index is not None:
                    if 0 <= a.card_index < len(hand):
                        hand.pop(a.card_index)
                    if rec.drew_order is not None:
                        hand.append(rec.drew_order)
                if a.type is ActionType.DISCARD and rec.discarded_card is not None:
                    discard_ranks.add(rec.discarded_card.rank)
                elif a.type is ActionType.PLAY and rec.success is False and rec.played_card:
                    discard_ranks.add(rec.played_card.rank)  # misplays land in discard
        return saved

    def _derive_called(self, obs: Observation, target: int) -> set:
        # A saved card is held, not played immediately, even if its rank plays now.
        return super()._derive_called(obs, target) - set(self._derive_saved_low(obs, target))

    def _choose_play(self, obs: Observation) -> Action | None:
        action = super()._choose_play(obs)
        if action is not None:
            return action
        saved = self._derive_saved_low(obs)
        if saved:
            lowest = min(obs.play_stacks.values())
            for i, view in enumerate(obs.own_hand):
                r = saved.get(view.order)
                if r is not None and lowest >= r - 1:  # its stack is now exactly r-1
                    return Action.play(i)
        return None

    def _will_play(self, obs: Observation) -> tuple[set, set]:
        orders, ids = super()._will_play(obs)
        for p in range(obs.num_players):
            saved = self._derive_saved_low(obs, p)
            for cv in obs.hands[p]:
                if cv.order in saved:
                    orders.add(cv.order)
                    if cv.card is not None:
                        ids.add((cv.card.color, cv.card.rank))
        return orders, ids

    # --- giving -----------------------------------------------------------
    def _critical_save(self, obs: Observation) -> Action | None:
        nxt = (obs.player_index + 1) % obs.num_players
        if nxt != obs.player_index and not self._player_has_play(obs, nxt):
            chop = self._chop_index(obs, nxt)
            if chop is not None:
                card = obs.hands[nxt][chop].card
                # Singleton number clue to a critical chop card (its discarded
                # duplicate makes (B) hold). Works even if the rank plays now.
                if (card.rank in (2, 3, 4) and self._is_critical(obs, card)
                        and sum(1 for c in obs.hands[nxt] if c.card.rank == card.rank) == 1):
                    return Action.clue_rank(nxt, card.rank)
        # Otherwise fall back to critsave's protective (multi / rank-unplayable) save.
        return super()._critical_save(obs)

    def _would_read_as_save(self, obs, p, act, touched) -> bool:
        return (
            act.type is ActionType.CLUE_RANK and act.rank in (2, 3, 4)
            and len(touched) == 1
            and self._rank_in_discard(obs, act.rank)
            and self._is_chop(obs, p, touched[0].order)
        )

    def _play_clue_for(self, obs, p, cv, colors, ranks):
        card = cv.card
        # A singleton number clue to the chop (with that rank in discard) reads as
        # a save: held, then played later (when all stacks reach r-1).
        if (card.rank in (2, 3, 4) and ranks.count(card.rank) == 1
                and self._rank_in_discard(obs, card.rank)
                and self._is_chop(obs, p, cv.order)):
            if colors.count(card.color) == 1:
                return Action.clue_color(p, card.color)  # plays immediately -> prefer
            # Otherwise give the number clue as a save. No dup risk and no need for
            # the card to be critical: it goes into `_will_play`, so its duplicate
            # is never play-clued, so its stack can't pass it before it plays.
            return Action.clue_rank(p, card.rank)
        return super()._play_clue_for(obs, p, cv, colors, ranks)

    def _clue_options(self, obs: Observation, p: int):
        return [
            (a, t) for (a, t) in FocusPlayer._clue_options(obs, p)
            if not self._would_read_as_save(obs, p, a, t)
        ]
