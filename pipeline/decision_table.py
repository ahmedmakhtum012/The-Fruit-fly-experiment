"""
Decision Table
---------------
Maps reward/aversive/ambiguity signals from the connectome sim
directly to Action enum values. This is where the fly's "personality"
actually lives -- not in the neuron firing patterns themselves, but
in how you interpret them into discrete choices.

Design:
  - R high, A low => LIKE (reward-dominant, clear signal)
  - R sustained across posts => FOLLOW (consistent reward signal)
  - A high, R low => NOOP (aversive-dominant, avoid engagement)
  - A very high (hard threshold) => REPLY (involuntary, raw reaction)
  - R and A both moderate => REPLY (ambivalent, confused reaction)
  - U crosses high threshold => SEARCH (doesn't know what this is)
  - Everything low => NOOP (indifferent)

Thresholds are tunable -- these are first-pass estimates that will
shift as you see real 48-hour behavior.
"""

from enum import Enum
from dataclasses import dataclass
# from typing import Optional


class Action(Enum):
    LIKE = 1
    REPLY = 2
    REPOST = 3
    QUOTE = 4
    FOLLOW = 5
    UNFOLLOW = 6
    BLOCK = 7
    POST = 8
    REPORT = 9
    DM = 10
    SEARCH = 11
    NOOP = 12


@dataclass
class Decision:
    action: Action
    confidence: float  # 0-1, how "sure" the signal was
    reasoning: str    # human-readable why this action fired


class DecisionTable:
    def __init__(self):
        # Thresholds -- tuned to observed signal ranges (R~0.08-0.12, A~0.05-0.07, U~0.6-0.8)
        self.REWARD_HIGH = 0.08
        self.REWARD_VERY_HIGH = 0.10
        self.AVERSIVE_HIGH = 0.05
        self.AVERSIVE_VERY_HIGH = 0.08
        self.AMBIGUITY_HIGH = 0.65
        self.AMBIGUITY_SEARCH = 0.80

    def decide(self, reward: float, aversive: float, ambiguity: float) -> Decision:
        """
        Takes R/A/U scalars from connectome sim, returns action + confidence.
        """
        r_dominant = reward > aversive + 0.02  # lower bar for reward dominance
        a_dominant = aversive > reward + 0.05
        both_high = reward > self.REWARD_HIGH and aversive > self.AVERSIVE_HIGH
        both_low = reward < 0.05 and aversive < 0.05

        # Rank decision priority by signal clarity + emotional intensity
        # AMBIGUITY CHECK FIRST -- "I don't understand this" overrides other signals

        if ambiguity >= self.AMBIGUITY_SEARCH:
            # Very confused, high ambiguity -> search to understand
            conf = min(ambiguity, 1.0)
            return Decision(
                action=Action.SEARCH,
                confidence=conf,
                reasoning=f"high ambiguity ({ambiguity:.3f}) -- what is this?"
            )

        if aversive > self.AVERSIVE_VERY_HIGH:
            # Very strong aversive signal -> involuntary reply, raw reaction
            conf = min(aversive, 1.0)
            return Decision(
                action=Action.REPLY,
                confidence=conf,
                reasoning=f"aversive spike ({aversive:.3f}) -- involuntary reaction"
            )

        # Check clear dominance BEFORE mixed signals
        if r_dominant and reward > self.REWARD_HIGH:
            # Clear reward signal, sufficient magnitude
            conf = reward
            return Decision(
                action=Action.LIKE,
                confidence=conf,
                reasoning=f"reward-dominant ({reward:.3f}) -- liked"
            )

        if a_dominant and aversive > self.AVERSIVE_HIGH:
            # Clear aversive signal, sufficient magnitude -> disengage
            conf = aversive
            return Decision(
                action=Action.NOOP,
                confidence=conf,
                reasoning=f"aversive-dominant ({aversive:.3f}) -- skip it"
            )

        if both_high:
            # Both reward and aversive lit up -> mixed feelings, ambivalent reply
            conf = (reward + aversive) / 2
            return Decision(
                action=Action.REPLY,
                confidence=conf,
                reasoning=f"mixed signals (R:{reward:.3f} A:{aversive:.3f}) -- conflicted"
            )

        if both_low:
            # Nothing triggered -> indifferent, move on
            return Decision(
                action=Action.NOOP,
                confidence=0.1,
                reasoning="flat signal -- no reaction"
            )

        # Fallback: weak/unclear signal -> do nothing
        return Decision(
            action=Action.NOOP,
            confidence=0.0,
            reasoning=f"unclear (R:{reward:.3f} A:{aversive:.3f} U:{ambiguity:.3f})"
        )

    def decide_follow(self, recent_likes: int) -> bool:
        """
        Heuristic: if the fly has liked several posts in a row from the
        same account, follow them. Tracks short-term reward consistency.
        Not signal-driven, just temporal pattern. Called externally by the
        main loop when tracking post source.
        """
        return recent_likes >= 3


if __name__ == "__main__":
    dt = DecisionTable()

    test_signals = [
        (0.20, 0.10, 0.30, "clear reward (kitten pic)"),
        (0.05, 0.45, 0.70, "clear aversive (bad news)"),
        (0.15, 0.15, 0.80, "high ambiguity (confusing meme)"),
        (0.60, 0.55, 0.50, "both high (controversial topic)"),
        (0.02, 0.03, 0.15, "flat (boring post)"),
    ]

    for r, a, u, desc in test_signals:
        dec = dt.decide(r, a, u)
        print(f"{desc:35} -> {dec.action.name:10} (conf: {dec.confidence:.2f})")
        print(f"  {dec.reasoning}")
        print()