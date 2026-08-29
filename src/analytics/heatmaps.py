"""Player/team positional heatmaps on a top-down pitch.

Needs pitch_x/pitch_y populated in the position log (i.e. a homography was
loaded during the pipeline run — see scripts/calibrate_pitch.py). Rows
without pitch coordinates are dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from mplsoccer import Pitch

from configs.field_config import PITCH_LENGTH, PITCH_WIDTH


def generate_heatmap(
    df: pd.DataFrame,
    out_path: str,
    track_id: int | None = None,
    team: int | None = None,
    title: str | None = None,
):
    data = df.dropna(subset=["pitch_x", "pitch_y"])
    data = data[data["class_name"] != "sports ball"]
    if track_id is not None:
        data = data[data["track_id"] == track_id]
    if team is not None:
        data = data[data["team"] == team]

    if data.empty:
        raise ValueError("No rows with pitch coordinates match the given filters")

    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH,
                  pitch_color="#22312b", line_color="#c7d5cc")
    fig, ax = pitch.draw(figsize=(10.5, 6.8))
    pitch.kdeplot(data["pitch_x"], data["pitch_y"], ax=ax, cmap="hot", fill=True,
                  levels=100, thresh=0.05, alpha=0.8)
    if title:
        ax.set_title(title, color="white", fontsize=14)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
