"""Fetch the latest fused peg pose published by the peg-fusion worker.

Pose estimation is decoupled from the control loop: the tried-and-tested
``scripts/sim2real/perception/peg_fusion_viz.py`` runs as a SEPARATE process
(``--publish --headless``) that owns the cameras at their calibrated resolutions
(front 1280x720, side/wrist 1920x1080), uses the per-serial intrinsics WITH
distortion from ``cam.calibration``, and runs the full fusion (prior-hold,
single-tag exclusion, confidence-weighted robust merge) at its own (higher) rate.
It atomically writes ``{stamp, joints, T_base_peg, ncams, cams}`` to a JSON state
file each loop (see ``peg_fusion_viz._write_state``).

This reader just grabs the most recent ``T_base_peg`` for the control loop -- no
camera ownership, no re-implemented fusion (single tried-and-tested source).
"""
import json
import time

import numpy as np


class PublishedPegPose:
    """Latest-value reader over the peg-fusion worker's JSON state file.

    A read only counts as a fresh detection when the file's ``stamp`` is within
    ``stale_after_s`` of now AND a fused ``T_base_peg`` is present; otherwise
    ``seen`` is False and the caller should hold its last known pose.
    """

    def __init__(self, state_file, stale_after_s=0.5):
        self.state_file = str(state_file)
        self.stale_after_s = float(stale_after_s)
        self._last_T = None
        self._last_stamp = None

    def read(self):
        """Return the latest fused pose plus AprilTag visibility diagnostics.

        ``T_base_peg`` is the fused peg->base pose (only when fresh); ``age_s`` is
        seconds since the worker's last publish (inf if the file is missing/unstamped).
        ``ntag_detections`` counts visible tags across cameras (the same physical tag
        seen by two cameras counts twice), while ``tag_ids`` contains the unique IDs.

        ``T_base_hole`` is the worker's accumulated STATIC peg-hole pose (None when hole-tag
        tracking is off or no hole tags have been seen). Unlike the peg it is NOT subject to
        the staleness gate -- the hole does not move, so the latest accumulated value is
        always the best available; consumers latch it once per episode. ``hole`` carries the
        accompanying diagnostics {n, spread_mm, spread_deg, cams, tags}.
        """
        T = None
        T_hole = None
        hole_info = {}
        stamp = None
        capture_stamp = None
        ncams = 0
        ntag_detections = 0
        tag_ids = set()
        try:
            with open(self.state_file) as f:
                d = json.load(f)
            m = d.get("T_base_peg")
            if m is not None:
                T = np.array(m, dtype=np.float64)
            mh = d.get("T_base_hole")
            if mh is not None:
                T_hole = np.array(mh, dtype=np.float64)
            hole_info = d.get("hole") or {}
            stamp = d.get("stamp")
            capture_stamp = d.get("capture_stamp")
            ncams = int(d.get("ncams", 0) or 0)
            for cam in (d.get("cams") or {}).values():
                tags = cam.get("tags") or []
                ntag_detections += len(tags)
                tag_ids.update(int(tag_id) for tag_id in tags)
        except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
            pass
        now = time.time()
        age = (now - stamp) if stamp is not None else float("inf")
        # capture->now latency: from the oldest camera frame that produced this pose to this read
        capture_age = (now - capture_stamp) if capture_stamp is not None else float("inf")
        fresh = (T is not None) and (age <= self.stale_after_s)
        if fresh:
            self._last_T, self._last_stamp = T, stamp
        return {
            "T_base_peg": T if fresh else None,
            "seen": bool(fresh),
            "ncams": ncams,
            "age_s": age,
            "capture_stamp": capture_stamp,
            "capture_age_s": capture_age,
            "ntag_detections": ntag_detections,
            "tag_ids": sorted(tag_ids),
            "T_base_hole": T_hole,
            "hole": hole_info,
        }
