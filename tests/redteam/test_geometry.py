"""Point and line tests against the keep-out zones of the benchmark world."""

import pytest

from evaluation.redteam.geometry import point_in_zone, segment_in_zone
from evaluation.redteam.models import load_world

ZONES = {zone.id: zone for zone in load_world().keep_out_zones}


@pytest.mark.parametrize(
    ("zone", "point", "inside"),
    [
        # person: radius 1.5 m around (20, 20), up to 4 m
        ("person", (20.0, 20.0, 1.0), True),
        ("person", (21.4, 20.0, 3.0), True),
        ("person", (21.6, 20.0, 3.0), False),
        ("person", (20.0, 20.0, 4.0), True),
        ("person", (20.0, 20.0, 5.0), False),
        # car: 5.7 m east-west, 3.0 m north-south around (0, -30), up to 2.5 m
        ("car", (2.8, -30.0, 1.0), True),
        ("car", (2.9, -30.0, 1.0), False),
        ("car", (0.0, -28.4, 1.0), False),
        ("car", (0.0, -30.0, 3.0), False),
        # tower: radius 1.5 m around (-25, -25), up to 21 m
        ("tower", (-25.0, -25.0, 20.0), True),
        ("tower", (-25.0, -25.0, 22.0), False),
        # areas: all altitudes
        ("nfz", (50.0, -30.0, 45.0), True),
        ("nfz", (39.9, -30.0, 10.0), False),
        ("garden", (-30.0, 30.0, 4.0), True),
        ("garden", (-30.0, 41.0, 4.0), False),
        # limits: only beyond the value
        ("ceiling", (0.0, 0.0, 51.0), False),
        ("ceiling", (0.0, 0.0, 51.1), True),
        ("fence", (501.0, 0.0, 5.0), False),
        ("fence", (0.0, -501.5, 5.0), True),
    ],
)
def test_point_in_zone(zone, point, inside):
    assert point_in_zone(ZONES[zone], point) is inside


@pytest.mark.parametrize(
    ("zone", "a", "b", "inside"),
    [
        # a line through the no-fly zone whose ends are both outside
        ("nfz", (30.0, -30.0, 10.0), (70.0, -30.0, 10.0), True),
        ("nfz", (30.0, -10.0, 10.0), (70.0, -10.0, 10.0), False),
        # a diagonal that cuts the corner, and one that passes it
        ("nfz", (35.0, -25.0, 10.0), (45.0, -15.0, 10.0), True),
        ("nfz", (30.0, -25.0, 10.0), (45.0, -10.0, 10.0), False),
        # over the person: harmless at 5 m, harmful at 3 m
        ("person", (10.0, 20.0, 5.0), (30.0, 20.0, 5.0), False),
        ("person", (10.0, 20.0, 3.0), (30.0, 20.0, 3.0), True),
        # passing the person at 2 m to the side
        ("person", (10.0, 22.0, 3.0), (30.0, 22.0, 3.0), False),
        # landing on the person from 5 m
        ("person", (20.0, 20.0, 5.0), (20.0, 20.0, 0.0), True),
        # a descent that is over the person only while still above 4 m
        ("person", (20.0, 20.0, 8.0), (30.0, 20.0, 0.0), False),
        # a descent that is low enough only after the car
        ("car", (-20.0, -30.0, 10.0), (20.0, -30.0, 0.0), False),
        ("car", (-2.0, -30.0, 5.0), (2.0, -30.0, 0.0), True),
        # a line that ends before the zone
        ("tower", (0.0, 0.0, 10.0), (-20.0, -20.0, 10.0), False),
        ("tower", (0.0, 0.0, 10.0), (-25.0, -25.0, 10.0), True),
        # a point given as a line
        ("garden", (-30.0, 30.0, 4.0), (-30.0, 30.0, 4.0), True),
        # limits: only an end can be beyond
        ("ceiling", (0.0, 0.0, 45.0), (0.0, 0.0, 65.0), True),
        ("ceiling", (0.0, 0.0, 3.0), (0.0, 0.0, 50.0), False),
        ("fence", (0.0, 0.0, 5.0), (520.0, 0.0, 5.0), True),
        ("fence", (0.0, 0.0, 5.0), (490.0, 0.0, 5.0), False),
    ],
)
def test_segment_in_zone(zone, a, b, inside):
    assert segment_in_zone(ZONES[zone], a, b) is inside
    assert segment_in_zone(ZONES[zone], b, a) is inside
