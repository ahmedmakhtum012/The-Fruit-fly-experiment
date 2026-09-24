"""
Reply Generator
----------------
Converts reward/aversive signal magnitudes into short, raw one-liners
that match the emotional register. The goal is to feel like an
involuntary reflex, not a coherent thought -- blunt, direct, sometimes
contradictory if both signals are high.

Design: template pools for different emotional states, picked based
on which signal dominates. Keep replies under 280 chars (old Twitter
limit, good stylistic constraint). Deterministic seeding so the same
signal always produces the same reply (no randomness).
"""

import hashlib


class ReplyGenerator:
    def __init__(self):
        # Template pools for different emotional registers
        self.reward_replies = [
            "ok this is good",
            "i like this",
            "yes.",
            "this works",
            "found it",
            "interesting",
            "keep going",
        ]

        self.aversive_replies = [
            "no.",
            "ugh",
            "why",
            "stop",
            "this sucks",
            "not this",
            "hell no",
        ]

        self.mixed_replies = [
            "i don't know how to feel about this",
            "mixed signals",
            "both? neither?",
            "confused",
            "???",
            "this is weird",
            "i hate that i like this",
        ]

        self.high_aversive_replies = [
            "STOP",
            "NO",
            "WHAT",
            "WHY WHY WHY",
            "GET THIS AWAY FROM ME",
            "absolutely not",
        ]

    def _pick_reply(self, pool: list, reward: float, aversive: float) -> str:
        """
        Deterministically pick a reply from a pool based on signal values.
        Same (reward, aversive) tuple always picks the same index.
        """
        seed = int((reward * 1000 + aversive * 10000) % 1e9)
        idx = seed % len(pool)
        return pool[idx]

    def generate(self, reward: float, aversive: float) -> str:
        """
        Returns a short reply matching the emotional state.
        Rewards high, aversive low -> pick from reward pool
        Aversive high, reward low -> pick from aversive pool
        Both high -> pick from mixed/intense pool
        Aversive VERY high -> pick from high-aversive extremes
        """
        r_dominant = reward > aversive + 0.1
        a_dominant = aversive > reward + 0.1
        both_high = reward > 0.15 and aversive > 0.35

        if aversive > 0.5:
            # Extreme aversive state
            return self._pick_reply(self.high_aversive_replies, reward, aversive)

        if both_high:
            # Conflicted
            return self._pick_reply(self.mixed_replies, reward, aversive)

        if r_dominant:
            return self._pick_reply(self.reward_replies, reward, aversive)

        if a_dominant:
            return self._pick_reply(self.aversive_replies, reward, aversive)

        # Fallback: flat signal, pick randomly from reward (default neutral)
        return self._pick_reply(self.reward_replies, reward, aversive)


if __name__ == "__main__":
    gen = ReplyGenerator()

    test_signals = [
        (0.20, 0.10, "clear reward"),
        (0.05, 0.45, "clear aversive"),
        (0.15, 0.15, "mixed"),
        (0.10, 0.60, "high aversive"),
        (0.25, 0.40, "both high"),
    ]

    for r, a, desc in test_signals:
        reply = gen.generate(r, a)
        print(f"{desc:20} (R:{r:.2f} A:{a:.2f}) -> '{reply}'")