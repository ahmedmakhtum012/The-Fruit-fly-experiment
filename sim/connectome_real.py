"""
Connectome Sim Engine (real data version)
-------------------------------------------
Loads the cached real neuron populations + weighted edges, runs a
leaky-integrate-and-fire pass driven by a stimulus, and reads out
reward/aversive/ambiguity signals.

Concentrated-first design: this only simulates the neurons that
actually matter for the readout -- i.e. sensory neurons that have at
least one direct edge into reward or aversive (299 + 97 = up to 396
relevant source neurons, not all 102,810). This isn't a shortcut on
the DATA (all edges come from the real fetched graph, nothing is
faked) -- it's a shortcut on which of the 102k sensory neurons get
touched by the sim loop, since neurons with zero outgoing edges to
our two output clusters cannot affect the readout at all, full stop,
regardless of how they're activated. Diluting further (subsampling,
adding intermediate hops) happens later only if this is too coarse
or too slow.
"""

import json
import hashlib
from pathlib import Path

import numpy as np

from stimulus_encoder import StimulusEncoder

CACHE_DIR = Path(__file__).resolve().parent.parent / "logs" / "neuron_cache"

# LIF parameters
LEAK = 0.15          # fraction of potential lost per timestep
THRESHOLD = 1.0       # spike threshold
STEPS = 20            # timesteps per stimulus pass
REFRACTORY_RESET = 0.0


class ConnectomeSim:
    def __init__(self):
        self._load_edges()
        self._build_index_maps()
        self.encoder = StimulusEncoder(n_sensory=len(self.sensory_ids))

    def _load_edges(self):
        def load(name):
            path = CACHE_DIR / f"edges_{name}.json"
            with open(path) as f:
                return json.load(f)

        self.s2r = load("sensory_to_reward")       # sensory -> reward
        self.s2a = load("sensory_to_aversive")      # sensory -> aversive
        self.r2a = load("reward_to_aversive")
        self.a2r = load("aversive_to_reward")

    def _build_index_maps(self):
        s_ids = sorted(set(
            [e["bodyId_pre"] for e in self.s2r] + [e["bodyId_pre"] for e in self.s2a]
        ))
        r_ids = sorted(set(
            [e["bodyId_post"] for e in self.s2r] + [e["bodyId_pre"] for e in self.r2a]
            + [e["bodyId_post"] for e in self.a2r]
        ))
        a_ids = sorted(set(
            [e["bodyId_post"] for e in self.s2a] + [e["bodyId_post"] for e in self.r2a]
            + [e["bodyId_pre"] for e in self.a2r]
        ))

        self.sensory_ids = s_ids
        self.reward_ids = r_ids
        self.aversive_ids = a_ids
        self.s_idx = {bid: i for i, bid in enumerate(s_ids)}
        self.r_idx = {bid: i for i, bid in enumerate(r_ids)}
        self.a_idx = {bid: i for i, bid in enumerate(a_ids)}

        self.W_s2r = np.zeros((len(s_ids), len(r_ids)))
        for e in self.s2r:
            self.W_s2r[self.s_idx[e["bodyId_pre"]], self.r_idx[e["bodyId_post"]]] += e["weight"]

        self.W_s2a = np.zeros((len(s_ids), len(a_ids)))
        for e in self.s2a:
            self.W_s2a[self.s_idx[e["bodyId_pre"]], self.a_idx[e["bodyId_post"]]] += e["weight"]

        self.W_r2a = np.zeros((len(r_ids), len(a_ids)))
        for e in self.r2a:
            self.W_r2a[self.r_idx[e["bodyId_pre"]], self.a_idx[e["bodyId_post"]]] += e["weight"]

        self.W_a2r = np.zeros((len(a_ids), len(r_ids)))
        for e in self.a2r:
            self.W_a2r[self.a_idx[e["bodyId_pre"]], self.r_idx[e["bodyId_post"]]] += e["weight"]

        for W in [self.W_s2r, self.W_s2a, self.W_r2a, self.W_a2r]:
            if W.max() > 0:
                W /= W.max()

        print(f"Sim graph: {len(s_ids)} active sensory, {len(r_ids)} reward, "
              f"{len(a_ids)} aversive neurons wired.")

    def _stimulus_to_sensory_drive(self, text: str = None, image_bytes: bytes = None) -> np.ndarray:
        return self.encoder.encode(text=text, image_bytes=image_bytes)

    def run(self, text: str = None, image_bytes: bytes = None) -> dict:
        drive = self._stimulus_to_sensory_drive(text=text, image_bytes=image_bytes)

        r_potential = np.zeros(len(self.reward_ids))
        a_potential = np.zeros(len(self.aversive_ids))
        r_spikes = np.zeros(len(self.reward_ids))
        a_spikes = np.zeros(len(self.aversive_ids))

        for t in range(STEPS):
            step_drive = drive * (1 - t / STEPS)

            r_input = step_drive @ self.W_s2r + a_spikes @ self.W_a2r
            a_input = step_drive @ self.W_s2a + r_spikes @ self.W_r2a

            r_potential = r_potential * (1 - LEAK) + r_input
            a_potential = a_potential * (1 - LEAK) + a_input

            r_spikes = (r_potential >= THRESHOLD).astype(float)
            a_spikes = (a_potential >= THRESHOLD).astype(float)

            r_potential[r_spikes.astype(bool)] = REFRACTORY_RESET
            a_potential[a_spikes.astype(bool)] = REFRACTORY_RESET

        # Compute scale factor to balance populations based on their mean input sensitivity
        # Reward: 277 neurons, Aversive: 16 neurons
        # From debug: r_input.mean() ≈ 0.09, a_input.mean() ≈ 0.44 -> ratio ~4.9
        scale_factor = 0.2  # 1/4.9 ≈ 0.2 to bring aversive down to reward scale
        reward_signal = float(np.clip(r_potential.mean() + r_spikes.mean(), 0, 1))
        aversive_signal = float(np.clip((a_potential.mean() + a_spikes.mean()) * scale_factor, 0, 1))
        # Ambiguity: low when one signal clearly dominates, high when both weak or both strong and close
        signal_max = max(reward_signal, aversive_signal)
        signal_diff = abs(reward_signal - aversive_signal)
        ambiguity_signal = float(1.0 - signal_diff / max(signal_max, 0.01))

        return {
            "reward": reward_signal,
            "aversive": aversive_signal,
            "ambiguity": ambiguity_signal,
        }


if __name__ == "__main__":
    sim = ConnectomeSim()
    test_stimuli = [
        "just adopted the cutest kitten ever look at this fluff",
        "google mapped the entire fruit fly connectome open source",
        "my flight got cancelled and I'm stuck at the airport for 8 hours",
    ]
    for text in test_stimuli:
        result = sim.run(text=text)
        print(f"{text[:45]!r:48} -> {result}")