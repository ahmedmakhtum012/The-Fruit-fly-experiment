"""
Stimulus Encoder (real features version)
-------------------------------------------
Replaces the hash-noise placeholder with actual extracted features
from real content, so different stimuli produce meaningfully
different sensory drive patterns instead of near-uniform noise.

Two paths depending on what the stimulus is:
  - Image (screenshot of a post): color/brightness/edge statistics
  - Text (a post's text content): simple bag-of-words style embedding

Both get projected down to the size of the active sensory population
(currently 65 neurons) via a fixed random projection matrix -- fixed
per-run (seeded), not per-stimulus, so the *mapping* stays consistent
across stimuli but the *content* driving it is real.
"""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import io


class StimulusEncoder:
    def __init__(self, n_sensory: int, seed: int = 7):
        self.n_sensory = n_sensory
        # Fixed random projection -- consistent mapping from raw feature
        # space to sensory neuron count, same every run (not per-stimulus)
        rng = np.random.default_rng(seed)
        self.image_feat_dim = 64   # color histogram bins + stats
        self.text_feat_dim = 128   # hashed bag-of-words dim
        self.image_proj = rng.normal(0, 1, (self.image_feat_dim, n_sensory))
        self.text_proj = rng.normal(0, 1, (self.text_feat_dim, n_sensory))

    def _extract_image_features(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((64, 64))  # cheap, consistent size
        arr = np.array(img).astype(np.float64) / 255.0

        feats = []
        # Per-channel color histograms (coarse, 8 bins each = 24 dims)
        for c in range(3):
            hist, _ = np.histogram(arr[:, :, c], bins=8, range=(0, 1))
            feats.extend(hist / hist.sum())

        # Brightness stats
        gray = arr.mean(axis=2)
        feats.append(gray.mean())
        feats.append(gray.std())

        # Crude edge density (gradient magnitude) -- proxy for "visual complexity"
        gx = np.abs(np.diff(gray, axis=0)).mean()
        gy = np.abs(np.diff(gray, axis=1)).mean()
        feats.append(gx)
        feats.append(gy)

        # Pad/trim to fixed dim
        feats = np.array(feats)
        if len(feats) < self.image_feat_dim:
            feats = np.pad(feats, (0, self.image_feat_dim - len(feats)))
        else:
            feats = feats[: self.image_feat_dim]
        return feats

    def _extract_text_features(self, text: str) -> np.ndarray:
        """
        Simple hashed bag-of-words: each word deterministically hashes
        into one of text_feat_dim buckets, bucket counts = feature vec.
        Crude, but real (content-dependent), unlike pure stimulus hashing.
        """
        vec = np.zeros(self.text_feat_dim)
        words = text.lower().split()
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            vec[h % self.text_feat_dim] += 1
        if vec.sum() > 0:
            vec /= vec.sum()
        return vec

    def encode(self, *, image_bytes: bytes = None, text: str = None) -> np.ndarray:
        """
        Returns a sensory drive vector of length n_sensory, in [0, 1].
        Provide image_bytes and/or text -- if both given, features are
        averaged after projection.
        """
        drives = []

        if image_bytes is not None:
            feats = self._extract_image_features(image_bytes)
            drive = feats @ self.image_proj
            drives.append(drive)

        if text is not None:
            feats = self._extract_text_features(text)
            drive = feats @ self.text_proj
            drives.append(drive)

        if not drives:
            raise ValueError("Must provide image_bytes and/or text")

        combined = np.mean(drives, axis=0)
        # Scale before squashing -- raw projections of sparse feature
        # vectors are small-magnitude, so sigmoid(raw) collapses toward
        # 0.5 regardless of real content differences. Scaling first
        # preserves separation between genuinely different stimuli.
        combined = combined * 8.0
        combined = 1 / (1 + np.exp(-combined))
        return combined


if __name__ == "__main__":
    enc = StimulusEncoder(n_sensory=65)

    # Text-only test -- three genuinely different posts
    for text in [
        "just adopted the cutest kitten ever look at this fluff",
        "google mapped the entire fruit fly connectome open source",
        "my flight got cancelled and I'm stuck at the airport for 8 hours",
    ]:
        drive = enc.encode(text=text)
        print(f"{text[:40]!r:45} -> mean={drive.mean():.3f} std={drive.std():.3f}")