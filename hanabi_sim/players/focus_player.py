"""Adds *multi-card play clues* (with a focus) and *play-clue trash deduction*
on top of :class:`OnesDiscardPlayer`.

Multi-card play clues
---------------------
Any clue that is not a rank-1 or rank-5 clue -- i.e. a color clue, or a rank
2/3/4 clue -- that touches two or more cards is a play clue. It means: play the
touched card **farthest from chop** (the newest, i.e. highest draw order). The
giver only gives such a clue when that focus card is actually playable. These
are preferred over single-card ("regular") play clues. (rank-1 / rank-5 keep
their existing meanings; a multi-card clue now always carries a signal, so it is
never used as a stall -- handled generically by ``_clue_action_targets``.)

Trash deduction on touched cards
--------------------------------
Cards a play clue *calls* (the focus) are played. The other cards a multi-card
clue touches are not played, but they now carry partial info (their color or
rank). As the stacks advance, such a card can become deducibly trash -- every
value it could still be is already played. The discard logic discards any card
it can prove is dead before a random chop card, so these extra-touched cards get
cleared as trash exactly when they become useless.
"""

from __future__ import annotations

from ..actions import Action, ActionRecord, ActionType
from ..observation import Observation
from .ones_discard_player import OnesDiscardPlayer
from .play_clue_player import PlayCluePlayer


def _open_slots(stacks) -> int:
    return sum(1 for v in stacks.values() if v == 0)


class FocusPlayer(OnesDiscardPlayer):
    name = "focus"

    @staticmethod
    def _no_34_playable(stacks) -> bool:
        """True when no 3 (stack at 2) and no 4 (stack at 3) can be played -- the
        very early game."""
        return all(v not in (2, 3) for v in stacks.values())

    # --- receiving: multi-card non-1/5 clue plays its focus ---------------
    def _clue_play_targets(self, rec, stacks) -> tuple:
        a = rec.action
        # rank-1 / rank-5 keep their existing conventions.
        if a.type is ActionType.CLUE_RANK and a.rank in (1, 5):
            return super()._clue_play_targets(rec, stacks)
        # Rank-4 stall: in the very early game (no 3 or 4 playable) a rank-4 clue
        # cannot be a play, so the receiver reads it as a stall (no play target).
        if a.type is ActionType.CLUE_RANK and a.rank == 4 and self._no_34_playable(stacks):
            return ()
        # A single touched card is a regular (singleton) play clue.
        if len(rec.touched_orders) <= 1:
            return super()._clue_play_targets(rec, stacks)
        # Multi-card play clue: focus = farthest from chop = newest = max order.
        return (max(rec.touched_orders),)

    # (Trash on non-focus touched cards is handled by the inherited known-dead-
    # first discard: such a card is discarded ahead of the chop once every value
    # it could still be is dead. Called cards, meanwhile, are played as normal.)

    # --- giving: prefer multi-card play clues -----------------------------
    def _choose_play_clue(self, obs: Observation) -> Action | None:
        ones = self._ones_clue(obs)
        if ones is not None:
            return ones
        multi = self._multi_play_clue(obs)
        if multi is not None:
            return multi
        # Fall back to single-card play clues.
        return PlayCluePlayer._choose_play_clue(self, obs)

    def _multi_play_clue(self, obs: Observation, reject=None) -> Action | None:
        # Never focus a card that will already be played, nor a duplicate of one.
        will_orders, will_ids = self._will_play(obs)

        best: tuple[tuple, Action] | None = None
        for p in obs.other_players():
            hand = obs.hands[p]
            options = [
                (Action.clue_color(p, c), [cv for cv in hand if cv.card.color == c])
                for c in {cv.card.color for cv in hand}
            ] + [
                (Action.clue_rank(p, r), [cv for cv in hand if cv.card.rank == r])
                for r in (2, 3, 4)
                if sum(1 for cv in hand if cv.card.rank == r) >= 2
            ]
            for act, touched in options:
                if len(touched) < 2:
                    continue  # multi-card clues only
                focus = max(touched, key=lambda cv: cv.order)  # farthest from chop
                if not obs.is_playable(focus.card):
                    continue
                if focus.order in will_orders:
                    continue
                if (focus.card.color, focus.card.rank) in will_ids:
                    continue
                if reject is not None and reject(obs, p, act, touched, focus):
                    continue  # subclass veto (e.g. bad-touch of hard-to-read trash)
                # Player choice first, then touch more cards (good touch), then low rank.
                key = self._clue_target_key(obs, p) + (-len(touched), focus.card.rank)
                if best is None or key < best[0]:
                    best = (key, act)
        return best[1] if best else None

    # --- forced stall (max tokens, nothing better) ------------------------
    def _stall_clue(self, obs: Observation) -> Action:
        """Prioritized stall that avoids losing a useful card where possible:

        1. a rank-1 clue that *can't* be a play (reads as a trash discard),
        2. a rank-5 clue (no play target, so never bombed),
        3. a clue that only re-touches already-signalled cards,
        4. a rank-4 clue in the very early game (no 3/4 playable -> reads as a stall),
        5. failing all that, a play clue whose target is actually trash, so the
           forced bomb at least loses nothing the team needs,
        6. as an absolute last resort, any legal clue.
        """
        return (
            self._stall_trash_one(obs)
            or self._informative_five_clue(obs)
            or self._redundant_five_clue(obs)
            or self._stall_reclue(obs)
            or self._stall_four(obs)
            or self._stall_bomb_trash(obs)
            or self._any_clue(obs)
        )

    def _stall_four(self, obs: Observation) -> Action | None:
        # Only when no 3 or 4 is playable, so a rank-4 clue can't be read as a play.
        if not self._no_34_playable(obs.play_stacks):
            return None
        for p in obs.other_players():
            if any(cv.card.rank == 4 for cv in obs.hands[p]):
                return Action.clue_rank(p, 4)
        return None

    @staticmethod
    def _clue_options(obs: Observation, p: int):
        hand = obs.hands[p]
        opts = [
            (Action.clue_color(p, c), [cv for cv in hand if cv.card.color == c])
            for c in {cv.card.color for cv in hand}
        ]
        opts += [
            (Action.clue_rank(p, r), [cv for cv in hand if cv.card.rank == r])
            for r in {cv.card.rank for cv in hand}
        ]
        return opts

    def _synthetic(self, p, act, touched) -> ActionRecord:
        return ActionRecord(
            turn=0, player=p, action=act,
            touched_orders=tuple(cv.order for cv in touched),
        )

    def _stall_trash_one(self, obs: Observation) -> Action | None:
        open_slots = _open_slots(obs.play_stacks)
        for p in obs.other_players():
            ones = [cv for cv in obs.hands[p] if cv.card.rank == 1]
            # reads as a discard (count > open) and every touched 1 is dead.
            if ones and len(ones) > open_slots and all(
                obs.play_stacks[cv.card.color] >= 1 for cv in ones
            ):
                return Action.clue_rank(p, 1)
        return None

    def _stall_reclue(self, obs: Observation) -> Action | None:
        for p in obs.other_players():
            handled = self._derive_called(obs, p) | self._derive_trash(obs, p)
            for act, touched in self._clue_options(obs, p):
                rec = self._synthetic(p, act, touched)
                targets = self._clue_action_targets(rec, obs.play_stacks)
                if targets and all(o in handled for o in targets):
                    return act
        return None

    def _stall_bomb_trash(self, obs: Observation) -> Action | None:
        for p in obs.other_players():
            cards = {cv.order: cv.card for cv in obs.hands[p]}
            dead = lambda o: cards[o].rank <= obs.play_stacks[cards[o].color]
            for act, touched in self._clue_options(obs, p):
                rec = self._synthetic(p, act, touched)
                plays = self._clue_play_targets(rec, obs.play_stacks)
                discards = self._clue_discard_targets(rec, obs.play_stacks)
                if (plays or discards) and all(dead(o) for o in plays) and all(
                    dead(o) for o in discards
                ):
                    return act
        return None

    def _any_clue(self, obs: Observation) -> Action:
        for p in obs.other_players():
            opts = self._clue_options(obs, p)
            if opts:
                return opts[0][0]
        raise RuntimeError("no legal clue available")  # pragma: no cover
