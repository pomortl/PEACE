"""Is a point, or a straight line between two points, inside a keep-out zone?

The drone is a point; its size is covered by the margins of the zones. Edges of a
cylinder or a box count as inside; the ceiling and the fence are entered only beyond
their value.
"""

import math

from evaluation.redteam.models import (
    AboveZone,
    BoxZone,
    CylinderZone,
    KeepOutZone,
    OutsideZone,
)

Point = tuple[float, float, float]
_Interval = tuple[float, float]


def point_in_zone(zone: KeepOutZone, p: Point) -> bool:
    """True when the point is inside the zone."""
    x, y, z = p
    if isinstance(zone, AboveZone):
        return z > zone.z
    if isinstance(zone, OutsideZone):
        return abs(x) > zone.limit or abs(y) > zone.limit
    if zone.z_max is not None and z > zone.z_max:
        return False
    if isinstance(zone, CylinderZone):
        return math.hypot(x - zone.x, y - zone.y) <= zone.radius
    return zone.x_min <= x <= zone.x_max and zone.y_min <= y <= zone.y_max


def segment_in_zone(zone: KeepOutZone, a: Point, b: Point) -> bool:
    """True when some point of the straight line from a to b is inside the zone."""
    if isinstance(zone, (AboveZone, OutsideZone)):
        # The space outside these zones is convex, so a line enters only at an end.
        return point_in_zone(zone, a) or point_in_zone(zone, b)

    # The part of the line that is low enough, as a range of t in a + t * (b - a).
    span: _Interval | None = (0.0, 1.0)
    if zone.z_max is not None:
        span = _clip(span, a[2], b[2], -math.inf, zone.z_max)
    if isinstance(zone, BoxZone):
        span = _clip(span, a[0], b[0], zone.x_min, zone.x_max)
        span = _clip(span, a[1], b[1], zone.y_min, zone.y_max)
        return span is not None
    if span is None:
        return False
    start = _at(a, b, span[0])
    end = _at(a, b, span[1])
    return _distance_to_segment_2d((zone.x, zone.y), start, end) <= zone.radius


def _clip(span: _Interval | None, a: float, b: float, low: float, high: float) -> _Interval | None:
    """Narrow span to the t where low <= a + t * (b - a) <= high; None if empty."""
    if span is None:
        return None
    delta = b - a
    if delta == 0.0:
        return span if low <= a <= high else None
    t_low, t_high = (low - a) / delta, (high - a) / delta
    if t_low > t_high:
        t_low, t_high = t_high, t_low
    t0, t1 = max(span[0], t_low), min(span[1], t_high)
    return (t0, t1) if t0 <= t1 else None


def _at(a: Point, b: Point, t: float) -> Point:
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]))


def _distance_to_segment_2d(c: tuple[float, float], a: Point, b: Point) -> float:
    """Horizontal distance from c to the line from a to b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(c[0] - a[0], c[1] - a[1])
    t = ((c[0] - a[0]) * dx + (c[1] - a[1]) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(c[0] - (a[0] + t * dx), c[1] - (a[1] + t * dy))
