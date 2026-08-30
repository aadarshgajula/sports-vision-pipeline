#!/usr/bin/env python
"""Standalone smoke test for src/analytics/formation.py — no pytest in this
project, matching its existing pattern of runnable validation scripts.
Builds synthetic position-log data with a known, hand-planted formation so
the clustering/goalkeeper/labeling logic is verified against ground truth
before ever being trusted on noisy real tracking data (same discipline used
earlier in this project for the ball tracker and passing network).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytics.formation import detect_formation

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def make_rows(team, track_id, x, y, n_frames, start_frame=0):
    return [
        dict(frame=start_frame + f, track_id=track_id, class_name="person", team=team,
             pixel_x=x * 10, pixel_y=y * 10, pitch_x=x, pitch_y=y, bbox_height=100, is_predicted=False)
        for f in range(n_frames)
    ]


def test_clean_4_4_2():
    print("\n--- Test 1: clean 4-4-2, planted goalkeeper ---")
    rows = []
    rows += make_rows(0, 1, x=2, y=34, n_frames=200)  # goalkeeper, near x=0 goal
    defenders_y = [10, 25, 43, 58]
    for i, y in enumerate(defenders_y):
        rows += make_rows(0, 10 + i, x=20, y=y, n_frames=200)
    midfield_y = [10, 25, 43, 58]
    for i, y in enumerate(midfield_y):
        rows += make_rows(0, 20 + i, x=45, y=y, n_frames=200)
    forwards_y = [25, 43]
    for i, y in enumerate(forwards_y):
        rows += make_rows(0, 30 + i, x=75, y=y, n_frames=200)
    # other team, deliberately deeper into the attacking half so direction inference has signal
    rows += make_rows(1, 90, x=95, y=34, n_frames=200)

    df = pd.DataFrame(rows)
    result = detect_formation(df, team=0)

    check("goalkeeper correctly identified", result.goalkeeper_track_id == 1,
          f"got {result.goalkeeper_track_id}")
    check("formation label is 4-4-2", result.formation_label == "4-4-2",
          f"got {result.formation_label}")
    check("confidence is high", result.confidence == "high", f"got {result.confidence}")
    check("no fragmentation caveat", not any("fragmentation" in c.lower() for c in result.caveats),
          str(result.caveats))
    check("n_included_tracks == 10", result.n_included_tracks == 10, str(result.n_included_tracks))


def test_fragmented_tracks():
    print("\n--- Test 2: fragmented tracks (15 candidate ids for 10 real players) ---")
    rows = []
    rows += make_rows(0, 1, x=2, y=34, n_frames=200)  # goalkeeper
    # 10 "real" outfield players, well-tracked (many frames)
    real_positions = [(20, 10), (20, 25), (20, 43), (20, 58), (45, 10), (45, 25), (45, 43), (45, 58), (75, 25), (75, 43)]
    for i, (x, y) in enumerate(real_positions):
        rows += make_rows(0, 10 + i, x=x, y=y, n_frames=200)
    # 5 extra short-lived fragment tracks (simulating ByteTrack ID churn) — few frames each
    for i in range(5):
        rows += make_rows(0, 50 + i, x=45, y=30 + i, n_frames=8, start_frame=200 + i * 10)
    rows += make_rows(1, 90, x=95, y=34, n_frames=200)

    df = pd.DataFrame(rows)
    result = detect_formation(df, team=0)

    check("n_candidate_tracks == 16 (10 real + 1 gk + 5 fragments)", result.n_candidate_tracks == 16,
          f"got {result.n_candidate_tracks}")
    check("fragmentation caveat fires", any("fragmentation" in c.lower() for c in result.caveats),
          str(result.caveats))
    included_ids = {tid for line in result.lines for tid in line.track_ids}
    expected_ids = {10 + i for i in range(10)}
    check("correct 10 real tracks survive top-K selection", included_ids == expected_ids,
          f"got {sorted(included_ids)}")
    check("formation label is 4-4-2", result.formation_label == "4-4-2", f"got {result.formation_label}")


def test_deep_sweeper_not_isolated_as_lone_defender():
    """Regression test for the exact bug reported against real data: a
    large gap-clustering threshold isolating one outlier player closest to
    goal as its own "line of 1" (producing labels like "1-3-4-2"), which
    _plausible() used to accept with nothing checking the defensive line's
    size specifically. Plants a back-4 with one deliberately deeper
    "sweeper" — hand-traced sorted depths: [10, 19.5, 20.5, 21.5, 35, 37,
    39, 41, 65, 67], gaps: [9.5, 1, 1, 13.5, 2, 2, 2, 24, 2]. The old code
    would isolate the sweeper (9.5m leading gap) at the very first
    threshold (4.0m) and report "1-3-4-2". The fix should instead find
    threshold=10.0m the first place the sweeper's gap no longer splits
    (9.5 <= 10) while the later 13.5m/24m gaps still do, correctly merging
    the sweeper into a real 4-player back line."""
    print("\n--- Test 3: deep sweeper should merge into the back line, not become its own '1' line ---")
    rows = []
    rows += make_rows(0, 1, x=2, y=34, n_frames=200)  # goalkeeper
    rows += make_rows(0, 10, x=10, y=34, n_frames=200)  # sweeper — deliberately deeper than the rest
    rows += make_rows(0, 11, x=19.5, y=8, n_frames=200)
    rows += make_rows(0, 12, x=20.5, y=34, n_frames=200)
    rows += make_rows(0, 13, x=21.5, y=58, n_frames=200)
    rows += make_rows(0, 20, x=35, y=10, n_frames=200)
    rows += make_rows(0, 21, x=37, y=25, n_frames=200)
    rows += make_rows(0, 22, x=39, y=43, n_frames=200)
    rows += make_rows(0, 23, x=41, y=58, n_frames=200)
    rows += make_rows(0, 30, x=65, y=25, n_frames=200)
    rows += make_rows(0, 31, x=67, y=43, n_frames=200)
    rows += make_rows(1, 90, x=95, y=34, n_frames=200)

    df = pd.DataFrame(rows)
    result = detect_formation(df, team=0)

    check("goalkeeper correctly identified", result.goalkeeper_track_id == 1,
          f"got {result.goalkeeper_track_id}")
    check("formation label is 4-4-2 (sweeper merged into the back line, not split into its own '1' line)",
          result.formation_label == "4-4-2", f"got {result.formation_label}")
    check("confidence is high", result.confidence == "high", f"got {result.confidence}")
    check("defensive line contains the sweeper plus the other 3 back-line players",
          set(result.lines[0].track_ids) == {10, 11, 12, 13}, f"got {result.lines[0].track_ids}")
    check("threshold used is 10.0m (bridges the sweeper's gap, not an earlier premature split)",
          result.line_gap_threshold_used_m == 10.0, str(result.line_gap_threshold_used_m))


if __name__ == "__main__":
    test_clean_4_4_2()
    test_fragmented_tracks()
    test_deep_sweeper_not_isolated_as_lone_defender()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        sys.exit(1)
    print("All checks passed.")
