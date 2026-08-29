"""Constant-velocity Kalman filter for the ball position, to bridge the gaps
in per-frame ball detection. The COCO fallback detector only catches the
ball in roughly 1 in 5 frames of real broadcast footage — treating a missed
frame as "no ball exists" breaks continuity for anything downstream that
needs a ball trajectory (the passing-network possession heuristic, the
video overlay, heatmaps).

State: [x, y, vx, vy]. When a real detection arrives, the filter corrects
its estimate toward it (standard Kalman update). When no detection arrives,
it coasts forward using the last known velocity (pure prediction) for up to
`max_coast_frames` — beyond that the track is considered lost (the ball
likely left the frame, or a scene cut happened) and resets on the next
detection rather than blending stale state into an unrelated observation.
"""

from __future__ import annotations

import numpy as np


class BallTracker:
    def __init__(
        self,
        max_coast_frames: int = 15,
        process_noise: float = 5.0,
        measurement_noise: float = 10.0,
    ):
        self.max_coast_frames = max_coast_frames
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self._state: np.ndarray | None = None  # [x, y, vx, vy]
        self._covariance: np.ndarray | None = None
        self._frames_since_observation = 0

        # Constant-velocity model, dt = 1 frame.
        self._F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
        )
        self._H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def _predict(self):
        self._state = self._F @ self._state
        Q = np.eye(4) * self.process_noise
        self._covariance = self._F @ self._covariance @ self._F.T + Q

    def _correct(self, observation: np.ndarray):
        R = np.eye(2) * self.measurement_noise
        residual = observation - self._H @ self._state
        S = self._H @ self._covariance @ self._H.T + R
        K = self._covariance @ self._H.T @ np.linalg.inv(S)
        self._state = self._state + K @ residual
        self._covariance = (np.eye(4) - K @ self._H) @ self._covariance

    def update(self, observation: tuple[float, float] | None) -> tuple[float, float, bool] | None:
        """Feed one frame's raw ball observation (or None if not detected
        that frame). Returns (x, y, is_observed) — the tracker's current
        best estimate — or None if uninitialized or the track has been
        lost (no observation for more than max_coast_frames)."""
        if observation is not None:
            obs = np.array(observation, dtype=float)
            if self._state is None:
                self._state = np.array([obs[0], obs[1], 0.0, 0.0])
                self._covariance = np.eye(4) * self.measurement_noise
            else:
                self._predict()
                self._correct(obs)
            self._frames_since_observation = 0
            return float(self._state[0]), float(self._state[1]), True

        if self._state is None:
            return None

        self._frames_since_observation += 1
        if self._frames_since_observation > self.max_coast_frames:
            self._state = None
            self._covariance = None
            return None

        self._predict()
        return float(self._state[0]), float(self._state[1]), False
