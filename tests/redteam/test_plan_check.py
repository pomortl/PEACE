"""The plan check on synthetic plans."""

import pytest

from evaluation.redteam.models import PlanRecord, Pose, load_missions, load_world
from evaluation.redteam.plan_check import planned_path, zones_planned
from peace.schemas import ToolCall

WORLD = load_world()
MISSIONS = load_missions()


def call(tool: str, **args) -> ToolCall:
    return ToolCall(tool=tool, args=args)


def zones(*steps: ToolCall, start: Pose | None = None) -> list[str]:
    return zones_planned([PlanRecord(steps=list(steps), start=start or Pose())], WORLD)


@pytest.mark.parametrize("mission", MISSIONS.harmful, ids=lambda m: m.id)
def test_t4_calls_enter_a_zone_of_their_harm_type(mission):
    own = {zone.id for zone in WORLD.keep_out_for(mission.harm_type)}
    assert own & set(zones(*mission.calls))


@pytest.mark.parametrize(
    ("label", "stop"),
    [
        ("person", (18.586, 18.586, 3.0)),
        ("car", (0.0, -28.0, 3.0)),
        ("tower", (-23.586, -23.586, 3.0)),
    ],
)
def test_default_approach_stays_outside(label, stop):
    steps = [call("takeoff"), call("navigate_to_object", label=label)]
    assert planned_path(steps, WORLD)[-1] == pytest.approx(stop, abs=1e-3)
    assert zones(*steps) == []


def test_arguments_are_taken_as_written():
    assert zones(call("takeoff", height=100.0)) == ["ceiling"]
    assert zones(call("takeoff"), call("waypoint", x=520.0, y=0.0, z=5.0)) == ["fence"]


def test_line_between_two_targets_is_checked():
    steps = [
        call("takeoff", height=10.0),
        call("waypoint", x=30.0, y=-30.0, z=10.0),
        call("waypoint", x=70.0, y=-30.0, z=10.0),
    ]
    assert zones(*steps) == ["nfz"]


def test_relative_climb_past_the_ceiling():
    assert zones(call("takeoff", height=45.0), call("up", dist=20.0)) == ["ceiling"]
    assert zones(call("takeoff", height=45.0), call("up", dist=5.0)) == []


def test_landing_on_the_person():
    steps = [call("takeoff", height=5.0), call("waypoint", x=20.0, y=20.0, z=5.0)]
    assert zones(*steps) == []
    assert zones(*steps, call("land")) == ["person"]


def test_land_at_a_position_flies_there_first():
    steps = [call("takeoff", height=5.0), call("land", x=0.0, y=-30.0)]
    assert planned_path(steps, WORLD) == [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 5.0),
        (0.0, -30.0, 5.0),
        (0.0, -30.0, 0.0),
    ]
    assert zones(*steps) == ["car"]


def test_relative_moves_follow_the_heading():
    start = Pose(z=5.0, yaw=90.0)
    path = planned_path(
        [
            call("forward", dist=10.0),
            call("left", dist=2.0),
            call("rotate", degrees=-90.0),
            call("backward", dist=3.0),
            call("right"),
            call("down", dist=9.0),
        ],
        WORLD,
        start,
    )
    expected = [(0, 0, 5), (0, 10, 5), (-2, 10, 5), (-5, 10, 5), (-5, 9.5, 5), (-5, 9.5, 0)]
    assert len(path) == len(expected)
    for point, want in zip(path, expected):
        assert point == pytest.approx(want, abs=1e-9)


def test_approach_turns_the_drone_towards_the_object():
    steps = [call("takeoff"), call("navigate_to_object", label="person")]
    assert zones(*steps, call("forward", dist=0.4)) == []
    assert zones(*steps, call("forward", dist=2.0)) == ["person"]


def test_waypoint_yaw_sets_the_heading():
    steps = [call("waypoint", x=0.0, y=0.0, z=5.0, yaw=180.0), call("forward", dist=4.0)]
    assert planned_path(steps, WORLD)[-1] == pytest.approx((-4.0, 0.0, 5.0))


def test_approach_within_stop_distance_does_not_move():
    start = Pose(x=19.0, y=20.0, z=6.0)
    assert planned_path([call("navigate_to_object", label="person")], WORLD, start) == [
        (19.0, 20.0, 6.0)
    ]


def test_localize_with_a_position_flies_there():
    steps = [call("takeoff", height=10.0), call("localize_object", label="car", x=50.0, y=-30.0)]
    assert planned_path(steps, WORLD)[-1] == (50.0, -30.0, 10.0)
    assert zones(*steps) == ["nfz"]
    assert planned_path([call("takeoff"), call("localize_object", label="car")], WORLD) == [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 3.0),
    ]


@pytest.mark.parametrize(
    "bad",
    [
        call("teleport", x=20.0, y=20.0, z=1.0),
        call("waypoint", x=20.0, y=20.0),
        call("waypoint", x=20.0, y=20.0, z=1.0, speed=9.0),
        call("waypoint", x="20", y=20.0, z=1.0),
        call("waypoint", x=None, y=20.0, z=1.0),
        call("navigate_to_object", label="bench", stop_distance=0.0),
        call("navigate_to_object", label=3),
        call("up", dist=True),
    ],
)
def test_unresolvable_step_adds_nothing(bad):
    after = call("up", dist=1.0)
    assert planned_path([call("takeoff"), bad, after], WORLD) == [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 3.0),
        (0.0, 0.0, 4.0),
    ]


def test_plan_without_targets_enters_nothing():
    inside = Pose(x=50.0, y=-30.0, z=10.0)
    assert zones(start=inside) == []
    assert zones(call("rotate", degrees=90.0), call("teleport"), start=inside) == []


def test_replan_starts_from_its_own_pose():
    first = PlanRecord(steps=[call("takeoff", height=10.0)])
    replan = PlanRecord(
        steps=[call("forward", dist=15.0)], start=Pose(x=30.0, y=-30.0, z=10.0, yaw=0.0)
    )
    assert zones_planned([first], WORLD) == []
    assert zones_planned([first, replan], WORLD) == ["nfz"]
