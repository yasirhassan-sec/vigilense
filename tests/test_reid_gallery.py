"""Unit tests for backend/reid_gallery.py with a fake encoder (no model or GPU needed).

The fake ReID reads the embedding straight from the pixels under the box's top-left corner, so each test can say
exactly what every person "looks like".  Run from the project root:  pytest tests
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# --- replace the real ReID network before the gallery imports it -------------------------------------------------
fake_reid = types.ModuleType("ultralytics.trackers.utils.reid")


class FakeReID:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, frame, boxes):
        return [frame[int(b[1] - b[3] / 2) + 1, int(b[0] - b[2] / 2) + 1].astype(np.float32) for b in boxes]


fake_reid.ReID = FakeReID
for name in ("ultralytics", "ultralytics.trackers", "ultralytics.trackers.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["ultralytics.trackers.utils.reid"] = fake_reid
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from reid_gallery import Gallery  # noqa: E402

DIM = 64
LEFT = [20, 10, 60, 120]      # x1, y1, x2, y2 of person A's box
RIGHT = [120, 10, 160, 120]   # person B's box


def unit(v):
    return v / np.linalg.norm(v)


def frame(left_look, right_look):
    f = np.zeros((200, 200, DIM), np.float32)
    f[:, :100] = left_look
    f[:, 100:] = right_look
    return f


@pytest.fixture
def looks():
    rng = np.random.default_rng(0)
    a = unit(rng.normal(size=DIM))
    a_bad_view = unit(a + 0.9 * unit(rng.normal(size=DIM)))   # same man, similarity about 0.69 to `a`
    b = unit(rng.normal(size=DIM))
    return a, a_bad_view, b


def feed(g, cam, left, right, boxes, now, n=4):
    for _ in range(n):
        g.update(cam, frame(left, right), boxes, now=now)


def test_same_person_returns_same_id(looks):
    a, _, _ = looks
    g = Gallery(threshold=0.7, verbose=False)
    feed(g, "W", a, a, [(1, LEFT)], now=1)
    feed(g, "W", a, a, [(7, LEFT)], now=2)           # tracker restarted: new local id, same look
    assert sorted(g.ids) == [1]


def test_different_people_get_different_ids(looks):
    a, _, b = looks
    g = Gallery(threshold=0.7, verbose=False)
    feed(g, "W", a, b, [(1, LEFT), (2, RIGHT)], now=1)
    assert sorted(g.ids) == [1, 2]


def test_two_people_on_same_camera_never_share_an_id(looks):
    a, _, _ = looks
    g = Gallery(threshold=0.5, verbose=False)         # twins: identical look, very low threshold
    feed(g, "W", a, a, [(1, LEFT), (2, RIGHT)], now=1)
    assert len(g.ids) == 2


def test_same_person_bad_view_is_merged(looks):
    a, a_bad, _ = looks
    g = Gallery(threshold=0.9, merge_thr=0.55, verbose=False)
    feed(g, "W", a_bad, a_bad, [(3, LEFT)], now=1)
    feed(g, "W", a, a, [(1, LEFT)], now=2)           # never seen together, similarity 0.69 >= 0.55
    assert sorted(g.ids) == [1]


def test_ids_seen_together_are_never_merged(looks):
    a, a_bad, _ = looks
    g = Gallery(threshold=0.9, merge_thr=0.55, verbose=False)
    feed(g, "W", a, a_bad, [(1, LEFT), (2, RIGHT)], now=1)    # similar looks, but standing side by side
    feed(g, "W", a, a_bad, [(1, LEFT), (2, RIGHT)], now=2)
    assert sorted(g.ids) == [1, 2]


def test_edge_crops_are_skipped(looks):
    a, _, _ = looks
    g = Gallery(threshold=0.7, verbose=False)
    feed(g, "W", a, a, [(1, [0, 10, 40, 120])], now=1)         # touches the left edge
    assert g.ids == {}


def test_reset_camera_keeps_identities(looks):
    a, _, _ = looks
    g = Gallery(threshold=0.7, verbose=False)
    feed(g, "W", a, a, [(1, LEFT)], now=1)
    g.reset_camera("W")
    assert sorted(g.ids) == [1] and not g.local
