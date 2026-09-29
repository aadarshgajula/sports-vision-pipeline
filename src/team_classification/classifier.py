"""Team classification: nearest-anchor jersey color match.

Unsupervised color clustering (K-means over hue histograms, then various
attempts to patch grass contamination, achromatic kits, and a referee
bucket) turned into an unbounded whack-a-mole: every fix for one kit
combination surfaced a new failure mode for another, because there was
never any ground truth to anchor to — only guessed structure.

The real fix is to stop guessing. The pipeline already requires one manual
calibration step per video (pitch homography). This reuses that same
one-time human step: the user clicks one example player from each team
(and optionally the referee) once, up front — see webapp/jobs.py. Team
assignment then becomes nearest-anchor color distance, not clustering.
Accuracy comes from the human-provided example, not clever feature
engineering, so the color descriptor itself is deliberately simple: mean
BGR of the torso crop.
"""

from __future__ import annotations

import cv2
import numpy as np

# On wide tactical-cam footage a player can be as little as 15-25px tall.
# At that scale, a person's silhouette (gaps around arms/legs, motion blur)
# means grass shows through *inside* even a tight torso crop, so a naive
# mean color is still mostly grass — verified directly against this
# project's own test footage. Excluding grass pixels (one well-defined,
# near-constant confound) before averaging is not a reintroduction of the
# old hue-clustering system — it's a data-cleaning step underneath the same
# simple supervised nearest-anchor match, applied identically to anchor
# crops and every runtime crop, since comparing a clean anchor against a
# dirty query would systematically bias matches toward whichever anchor
# happens to sit closer to grass-mixed color.
GRASS_HUE_RANGE = (35, 95)
GRASS_SAT_MIN = 60


def crop_torso(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Crop the jersey region of a player bounding box: vertically between
    the neckline and waist, full width. Excludes the head (hair/skin tone
    shouldn't drive team clustering) and legs/shoes/background grass (which
    dilute the kit-color signal the classifier actually needs)."""
    x1, y1, x2, y2 = xyxy.astype(int)
    height = y2 - y1
    torso_y1 = y1 + int(height * 0.15)
    torso_y2 = y1 + int(height * 0.55)
    crop = frame[torso_y1:torso_y2, x1:x2]
    if crop.size == 0:
        return frame[y1:y2, x1:x2]  # bbox too small for a torso band; fall back to full box
    return crop


def _non_grass_pixels(crop_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0].astype(float), hsv[:, :, 1].astype(float)
    lo, hi = GRASS_HUE_RANGE
    is_grass = (hue >= lo) & (hue <= hi) & (sat >= GRASS_SAT_MIN)
    pixels = crop_bgr.reshape(-1, 3)
    keep = ~is_grass.reshape(-1)
    # If almost everything got excluded (a real green-kitted team, or a
    # crop that's pure grass because detection missed), fall back to every
    # pixel rather than averaging 2-3 leftover ones.
    if keep.sum() < max(3, 0.05 * len(pixels)):
        return pixels
    return pixels[keep]


def _mean_color(crop_bgr: np.ndarray) -> np.ndarray:
    return _non_grass_pixels(crop_bgr).mean(axis=0)


class TeamClassifier:
    def __init__(self):
        self._anchors: dict[int | None, np.ndarray] = {}
        self._fitted = False

    def fit_from_anchors(self, anchor_crops: dict[int | None, np.ndarray]):
        """anchor_crops: {0: crop, 1: crop, None: crop (referee, optional)} —
        one example torso crop per category, from the one-time calibration
        step. None is the referee/other bucket, matching the position log's
        existing team=None convention for "not on a team."."""
        if 0 not in anchor_crops or 1 not in anchor_crops:
            raise ValueError("Need at least a team 0 and team 1 anchor crop")
        self._anchors = {label: _mean_color(crop) for label, crop in anchor_crops.items()}
        self._fitted = True

    def predict(self, crops: list[np.ndarray]) -> list[int | None]:
        if not self._fitted:
            raise RuntimeError("TeamClassifier.fit_from_anchors() must be called before predict()")
        labels = list(self._anchors.keys())
        anchor_matrix = np.stack([self._anchors[label] for label in labels])
        colors = np.stack([_mean_color(crop) for crop in crops])
        dists = np.linalg.norm(colors[:, None, :] - anchor_matrix[None, :, :], axis=2)
        nearest = dists.argmin(axis=1)
        return [labels[i] for i in nearest]
