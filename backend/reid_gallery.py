"""
Shared re-identification gallery for VigiLense.

ByteTrack gives every person a LOCAL id per camera, and starts a new id when a person leaves the picture and comes back.
This gallery gives each real person one GLOBAL id (G1, G2, ...) by comparing what people look like:

  1. For every tracked person it collects a few appearance embeddings (a vector of numbers from a ReID network).
  2. Once a local track has `min_samples` good crops, its average embedding is compared with every person already in
     the gallery (cosine similarity). A match above `threshold` reuses that G id, otherwise a new G id is created.
  3. The gallery is shared by all cameras, so the same person on another camera gets the same G id.
  4. Safety rule: two people visible on the same camera at the same moment never get the same G id.
  5. Merging: a person first seen from a bad angle can get a second G id. Two G ids that have been seen on a common
     camera, were NEVER visible together at the same moment, and look alike (similarity >= merge_thr, lower than the
     match threshold) are merged into the lower id. "Never together" is the safety check: one man cannot stand
     next to himself, so two ids that were seen together are always kept apart.

Crops cut by the left/right picture edge, or very small ones, are not used (they give poor embeddings).
Use test_reid.py first to see how well the model separates the people in your own videos, then tune --reid-thr.
"""
import time

import numpy as np
from ultralytics.trackers.utils.reid import ReID


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


class Gallery:
    def __init__(self, model="yolo26n-reid.onnx", device=None, threshold=0.75, max_age=600.0,
                 min_samples=3, embed_every=3, edge=6, ema=0.9, verbose=True, merge_thr=None):
        if isinstance(device, int) or str(device).isdigit():
            device = f"cuda:{int(device)}"
        self.reid = ReID(model, device=device)
        self.threshold, self.max_age = threshold, max_age
        self.min_samples, self.embed_every, self.edge, self.ema = min_samples, embed_every, edge, ema
        self.verbose = verbose
        self.merge_thr = threshold - 0.07 if merge_thr is None else merge_thr
        self.cannot = set()  # frozenset({a, b}): ids that were visible together, so they are different people
        self.cams = {}       # global id -> set of cameras it was seen on
        self.ids = {}        # global id -> {"emb": unit vector, "seen": time}
        self.local = {}      # (camera, local track id) -> {"gid", "samples", "n", "seen"}
        self.next_gid = 1

    def reset_camera(self, cam):
        """Forget the local tracks of one camera (its tracker was restarted). The global identities stay."""
        for k in [k for k in self.local if k[0] == cam]:
            del self.local[k]

    def update(self, cam, frame, tracked, now=None):
        """tracked: [(local track id, [x1, y1, x2, y2]), ...] for one camera. Returns {local id: global id or None}."""
        now = time.time() if now is None else now
        H, W = frame.shape[:2]
        live = {tid for tid, _ in tracked}
        for k in [k for k, r in self.local.items() if k[0] == cam and k[1] not in live and now - r["seen"] > 60]:
            del self.local[k]
        for gid in [g for g, i in self.ids.items() if now - i["seen"] > self.max_age]:
            del self.ids[gid]
        for rec in self.local.values():                      # a purged identity must be matched again
            if rec["gid"] is not None and rec["gid"] not in self.ids:
                rec["gid"], rec["samples"] = None, []

        todo = []
        for tid, (x1, y1, x2, y2) in tracked:
            rec = self.local.setdefault((cam, tid), {"gid": None, "samples": [], "n": 0, "seen": now})
            rec["seen"], rec["n"] = now, rec["n"] + 1
            usable = x1 > self.edge and x2 < W - self.edge and (x2 - x1) >= 20 and (y2 - y1) >= 40
            if usable and (rec["gid"] is None or rec["n"] % self.embed_every == 0):
                todo.append((tid, rec, [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]))
        if todo:
            embs = self.reid(frame, np.array([b for _, _, b in todo], dtype=np.float32))
            for (tid, rec, _), e in zip(todo, embs):
                if e is None:
                    continue
                e = _unit(np.asarray(e, dtype=np.float32).ravel())
                if rec["gid"] is None:
                    rec["samples"] = (rec["samples"] + [e])[-10:]
                else:                                        # keep the stored look fresh, but only from crops that still match
                    ident = self.ids[rec["gid"]]
                    if float(e @ ident["emb"]) >= self.threshold - 0.1:
                        ident["emb"] = _unit(self.ema * ident["emb"] + (1 - self.ema) * e)

        for tid, _ in tracked:
            rec = self.local[(cam, tid)]
            if rec["gid"] is None and len(rec["samples"]) >= self.min_samples:
                mean = _unit(np.mean(rec["samples"], axis=0))
                taken = {r["gid"] for (c, t), r in self.local.items() if c == cam and t in live and r["gid"] is not None}
                best, sim = None, -1.0
                for gid, ident in self.ids.items():
                    if gid in taken:                         # already used by someone else visible on this camera
                        continue
                    s = float(mean @ ident["emb"])
                    if s > sim:
                        best, sim = gid, s
                if best is not None and sim >= self.threshold:
                    rec["gid"] = best
                    if self.verbose:
                        print(f"[REID] {cam}: track {tid} matched G{best} (similarity {sim:.2f})")
                else:
                    rec["gid"] = self.next_gid
                    self.ids[rec["gid"]] = {"emb": mean, "seen": now}
                    self.next_gid += 1
                    if self.verbose:
                        print(f"[REID] {cam}: track {tid} is new -> G{rec['gid']} "
                              f"(best similarity to others {sim:.2f}, needs {self.threshold:.2f})")
            if rec["gid"] is not None:
                self.ids[rec["gid"]]["seen"] = now
        live_g = sorted({self.local[(cam, tid)]["gid"] for tid, _ in tracked} - {None})
        for g in live_g:
            self.cams.setdefault(g, set()).add(cam)
        for i, a in enumerate(live_g):
            for b in live_g[i + 1:]:
                self.cannot.add(frozenset((a, b)))
        self._merge_similar()
        return {tid: self.local[(cam, tid)]["gid"] for tid, _ in tracked}

    def _merge_similar(self):
        """Merge pairs of ids that were never visible together, share a camera and look alike."""
        changed = True
        while changed:
            changed = False
            gids = sorted(self.ids)
            for i, a in enumerate(gids):
                for b in gids[i + 1:]:
                    if frozenset((a, b)) in self.cannot:
                        continue
                    if not (self.cams.get(a, set()) & self.cams.get(b, set())):
                        continue
                    s = float(self.ids[a]["emb"] @ self.ids[b]["emb"])
                    if s < self.merge_thr:
                        continue
                    self.ids[a]["emb"] = _unit(self.ids[a]["emb"] + self.ids[b]["emb"])
                    self.ids[a]["seen"] = max(self.ids[a]["seen"], self.ids[b]["seen"])
                    del self.ids[b]
                    self.cams[a] = self.cams.get(a, set()) | self.cams.pop(b, set())
                    for rec in self.local.values():
                        if rec["gid"] == b:
                            rec["gid"] = a
                    for pair in [p for p in self.cannot if b in p]:
                        self.cannot.discard(pair)
                        (other,) = pair - {b}
                        self.cannot.add(frozenset((a, other)))
                    if self.verbose:
                        print(f"[REID] merged G{b} into G{a} (similarity {s:.2f}, never seen together)")
                    changed = True
                    break
                if changed:
                    break
