"""Ego-progress (EP) reference for the live PDMS verifier.
"""
import numpy as np


class EgoProgressReference:
    """Route polyline + human-pace lookup backing the EP soft term.
        route_xy : (M, 2) array
            Logged ego route in the same world frame as the verifier's
            ``planned_traj`` (for HUGSIM: x = raw z, y = -raw x of the camtoworld
            translations, matching the ``ego_box`` remap).
        frame_dt : float
            Seconds between consecutive ``route_xy`` poses (infer from the scene's
            per-frame timestamps; do not assume a rate).
        progress_threshold : float
            Achievable-progress floor in meters below which EP is 1.0.
    """

    def __init__(self, route_xy, frame_dt: float, progress_threshold: float = 5.0):
        route_xy = np.asarray(route_xy, dtype=np.float64).reshape(-1, 2)
        self._xy = route_xy
        segments = np.diff(route_xy, axis=0)
        seg_len = np.linalg.norm(segments, axis=1)
        self._stations = np.concatenate([[0.0], np.cumsum(seg_len)])
        self._frame_dt = float(frame_dt)
        self._threshold = float(progress_threshold)

    def station(self, point) -> float:
        """Arc-length station (meters from route start) of ``point``'s closest
        projection onto the route polyline."""
        p = np.asarray(point, dtype=np.float64)
        a, b = self._xy[:-1], self._xy[1:]
        ab = b - a
        seg_len2 = np.einsum("ij,ij->i", ab, ab)
        t = np.einsum("ij,ij->i", p[None, :] - a, ab) / np.maximum(seg_len2, 1e-12)
        t = np.clip(np.where(seg_len2 > 0, t, 0.0), 0.0, 1.0)
        proj = a + t[:, None] * ab
        d2 = np.einsum("ij,ij->i", p[None, :] - proj, p[None, :] - proj)
        i = int(np.argmin(d2))
        return float(self._stations[i] + t[i] * np.sqrt(seg_len2[i]))

    def _nearest_vertex(self, point) -> int:
        p = np.asarray(point, dtype=np.float64)
        d2 = np.einsum("ij,ij->i", self._xy - p, self._xy - p)
        return int(np.argmin(d2))

    def expected_progress(self, point, horizon_s: float) -> float:
        """Meters the logged human advanced over ``horizon_s`` seconds starting
        from their pose nearest to ``point``.

        Saturates at the end of the log, so near the route end the achievable
        progress shrinks and the escape hatch takes over.
        """
        j = self._nearest_vertex(point)
        k = max(1, int(round(horizon_s / self._frame_dt)))
        end = min(j + k, len(self._stations) - 1)
        return float(self._stations[end] - self._stations[j])

    def ego_progress(self, planned_xy, horizon_s: float) -> float:
        """EP in [0, 1] for a global plan polyline over ``horizon_s`` seconds.
        """
        planned_xy = np.asarray(planned_xy, dtype=np.float64)
        if len(self._xy) < 2 or planned_xy.shape[0] < 2:
            return 1.0
        progress = max(self.station(planned_xy[-1]) - self.station(planned_xy[0]), 0.0)
        expected = self.expected_progress(planned_xy[0], horizon_s)
        if expected < self._threshold:
            return 1.0
        return float(np.clip(progress / expected, 0.0, 1.0))
