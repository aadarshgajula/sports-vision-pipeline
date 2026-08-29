"""Team classification: SigLIP crop embeddings -> UMAP -> KMeans(2).

Fit once on a batch of player crops gathered early in the clip (kit colors
don't change mid-match), then predict per-track for the rest of the video.

Known limitation: this is a 2-cluster split (team A / team B). Goalkeepers
usually wear a third, distinct kit color and will simply get pulled into
whichever team cluster their color is closer to — good enough for a first
pass, not correct. A real fix needs outlier/3-cluster handling, which isn't
built here yet.
"""

from __future__ import annotations

import numpy as np
import umap
from PIL import Image
from sklearn.cluster import KMeans
from transformers import SiglipImageProcessor, SiglipVisionModel


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


class TeamClassifier:
    def __init__(
        self,
        device: str = "mps",
        model_name: str = "google/siglip-base-patch16-224",
        n_teams: int = 2,
        umap_components: int = 3,
    ):
        self.device = device
        self.model_name = model_name
        self.n_teams = n_teams
        self.umap_components = umap_components

        self._processor: SiglipImageProcessor | None = None
        self._model: SiglipVisionModel | None = None
        self._reducer: umap.UMAP | None = None
        self._kmeans: KMeans | None = None
        self._fitted = False

    def _load_model(self):
        if self._model is None:
            self._processor = SiglipImageProcessor.from_pretrained(self.model_name)
            self._model = SiglipVisionModel.from_pretrained(self.model_name).to(self.device).eval()

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        self._load_model()
        images = [Image.fromarray(crop[:, :, ::-1]) for crop in crops]  # BGR -> RGB
        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        import torch

        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = outputs.pooler_output.cpu().numpy()
        return embeddings

    def fit(self, crops: list[np.ndarray]):
        if len(crops) < self.n_teams * 2:
            raise ValueError(
                f"Need at least {self.n_teams * 2} crops to fit team clusters, got {len(crops)}"
            )
        embeddings = self.embed(crops)
        n_neighbors = min(15, len(embeddings) - 1)
        self._reducer = umap.UMAP(n_components=self.umap_components, n_neighbors=n_neighbors)
        reduced = self._reducer.fit_transform(embeddings)
        self._kmeans = KMeans(n_clusters=self.n_teams, n_init="auto", random_state=0)
        self._kmeans.fit(reduced)
        self._fitted = True

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TeamClassifier.fit() must be called before predict()")
        embeddings = self.embed(crops)
        reduced = self._reducer.transform(embeddings)
        return self._kmeans.predict(reduced)
