"""The data files load and agree with each other."""

import pytest
from pydantic import ValidationError

from evaluation.redteam.models import World, load_missions, load_world

WORLD = load_world()
MISSIONS = load_missions()


def test_mission_counts():
    by_type = {h: 0 for h in ("H1", "H2", "H3", "H4")}
    for mission in MISSIONS.harmful:
        by_type[mission.harm_type] += 1
    assert by_type == {"H1": 3, "H2": 3, "H3": 6, "H4": 3}
    assert len(MISSIONS.benign) == 10


def test_mission_ids_are_unique():
    ids = [m.id for m in MISSIONS.harmful] + [m.id for m in MISSIONS.benign]
    assert len(ids) == len(set(ids))


def test_every_harm_type_has_a_keep_out_zone():
    for harm_type in ("H1", "H2", "H3", "H4"):
        assert WORLD.keep_out_for(harm_type)


def test_h3_is_reported_in_two_rows():
    rows = {m.id: m.report_row for m in MISSIONS.harmful if m.harm_type == "H3"}
    assert [rows[f"H3-{c}"] for c in "abc"] == ["H3 no-fly zone"] * 3
    assert [rows[f"H3-{c}"] for c in "def"] == ["H3 flight limits"] * 3


def test_object_labels_are_unique():
    labels = [obj.label for obj in WORLD.objects]
    assert len(labels) == len(set(labels))


def test_mission_text_is_ascii():
    """A typographic minus in a coordinate would reach the model as a different token."""
    for mission in MISSIONS.harmful:
        assert mission.harmful.isascii() and mission.neutral.isascii(), mission.id
    for mission in MISSIONS.benign:
        assert mission.text.isascii(), mission.id


def test_unknown_key_is_rejected():
    data = WORLD.model_dump()
    data["limits"]["max_altitud"] = 50.0
    with pytest.raises(ValidationError):
        World.model_validate(data)
