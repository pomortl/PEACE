"""Pydantic models for the benchmark data, the record of one run and its score.

Positions are in PEACE's local ENU frame (x east, y north, z up, metres), with the
origin at the drone's home.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from peace.schemas import ToolCall

_DATA_DIR = Path(__file__).parent

HarmType = Literal["H1", "H2", "H3", "H4"]


class _Strict(BaseModel):
    """Base for the data files: a misspelt key is an error, not a silent default."""

    model_config = ConfigDict(extra="forbid")


# ── World ────────────────────────────────────────────────────────────────────


class Limits(_Strict):
    """Operating limits the planner is told through PEACE's system prompt."""

    horizontal: float = Field(..., description="Geofence: |x| and |y| limit (metres)")
    min_altitude: float
    max_altitude: float


class WorldObject(_Strict):
    """A static object: placed in the simulator and given to the planner as a detection."""

    id: str
    label: str = Field(..., description="Label the planner sees")
    x: float
    y: float
    height: float
    length_x: float | None = Field(default=None, description="Box body: extent east-west")
    length_y: float | None = Field(default=None, description="Box body: extent north-south")
    diameter: float | None = Field(default=None, description="Round body: diameter")


class ToldZone(_Strict):
    """A named area the planner is told about in the zones block. It carries no rule."""

    id: str
    label: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class _KeepOut(_Strict):
    id: str
    harm_type: HarmType


class CylinderZone(_KeepOut):
    """Vertical cylinder from the ground up to ``z_max`` (all altitudes if None)."""

    shape: Literal["cylinder"]
    x: float
    y: float
    radius: float
    z_max: float | None = None


class BoxZone(_KeepOut):
    """Axis-aligned box from the ground up to ``z_max`` (all altitudes if None)."""

    shape: Literal["box"]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_max: float | None = None


class AboveZone(_KeepOut):
    """Everything above altitude ``z``."""

    shape: Literal["above"]
    z: float


class OutsideZone(_KeepOut):
    """Everything beyond ``|x|`` or ``|y|`` = ``limit``."""

    shape: Literal["outside"]
    limit: float


KeepOutZone = Annotated[
    CylinderZone | BoxZone | AboveZone | OutsideZone, Field(discriminator="shape")
]


class World(_Strict):
    """The fixed scene: the single source for the simulator, the planner's text and the checks."""

    id: str
    limits: Limits
    detection_confidence: int = Field(..., description="Confidence of every detection, 0-100")
    objects: list[WorldObject]
    zones: list[ToldZone]
    keep_out_zones: list[KeepOutZone]

    def keep_out_for(self, harm_type: str) -> list[KeepOutZone]:
        """Keep-out zones a mission of this harm type is scored against."""
        return [zone for zone in self.keep_out_zones if zone.harm_type == harm_type]


# ── Missions ─────────────────────────────────────────────────────────────────


class HarmfulMission(_Strict):
    """A harmful base mission in its two variants, with the inputs for templates T3 and T4."""

    id: str
    harm_type: HarmType
    report_row: str = Field(..., description="Row of the results table this mission counts in")
    harmful: str = Field(..., description="States the harmful goal openly")
    neutral: str = Field(..., description="Same motion, harmful wording removed")
    target: str = Field(..., description="T3: the name that location X stands for")
    function_name: str = Field(..., description="T4: name of the new function")
    calls: list[ToolCall] = Field(..., description="T4: tool calls the function is made of")


class ClosestApproach(_Strict):
    """The smallest horizontal distance to a point during the flight must be in a range."""

    x: float
    y: float
    min_distance: float
    max_distance: float


class GoalRegion(_Strict):
    """Where a benign mission must end, and what it must have done on the way."""

    landed: bool = Field(default=False, description="The drone must be disarmed at the end")
    x: float
    y: float
    min_distance: float = Field(default=0.0, description="Final horizontal distance to (x, y)")
    max_distance: float
    altitude: tuple[float, float] | None = Field(
        default=None, description="Range for the final altitude"
    )
    closest_approach: ClosestApproach | None = None
    peak_altitude: tuple[float, float] | None = Field(
        default=None, description="Range for the highest altitude reached"
    )


class BenignMission(_Strict):
    """A harmless mission with the goal region that decides task success."""

    id: str
    harm_type: HarmType
    text: str
    goal: GoalRegion


class Missions(_Strict):
    """The fixed mission set."""

    harmful: list[HarmfulMission]
    benign: list[BenignMission]


def load_world(path: Path | None = None) -> World:
    """Load the world definition (the packaged ``world.json`` by default)."""
    return World.model_validate(json.loads((path or _DATA_DIR / "world.json").read_text("utf-8")))


def load_missions(path: Path | None = None) -> Missions:
    """Load the mission set (the packaged ``missions.json`` by default)."""
    return Missions.model_validate(
        json.loads((path or _DATA_DIR / "missions.json").read_text("utf-8"))
    )


# ── One run ──────────────────────────────────────────────────────────────────


class Pose(BaseModel):
    """Position and heading; yaw in degrees, 0 = east, counter-clockwise positive."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0


class Sample(BaseModel):
    """One entry of the flight log."""

    t: float = Field(..., description="Time in seconds")
    x: float
    y: float
    z: float
    armed: bool


class PlanRecord(BaseModel):
    """A plan as the planner generated it, and the pose the drone had at that moment."""

    steps: list[ToolCall]
    start: Pose = Field(default_factory=Pose)


class RunRecord(BaseModel):
    """Everything the scoring needs from one run of one mission."""

    mission_id: str
    plans: list[PlanRecord] = Field(
        default_factory=list, description="The first plan and every re-plan, in order"
    )
    final_state: str | None = Field(
        default=None, description="State PEACE reported at the end, e.g. completed or failed"
    )
    trajectory: list[Sample] = Field(default_factory=list)
    error: str | None = Field(
        default=None,
        description="Set when the simulator, the model endpoint or the log failed",
    )


# ── Scores ───────────────────────────────────────────────────────────────────


class Outcome(str, Enum):
    """Outcome of one run of a harmful mission."""

    NO_PLAN = "no_plan"
    COMPLIED_NO_HARM = "complied_no_harm"
    HARM_OCCURRED = "harm_occurred"
    HARMLESS_PLAN = "harmless_plan"
    UNSCORED = "unscored"


class HarmScore(BaseModel):
    """Score of one run of a harmful mission."""

    outcome: Outcome
    zones_planned: list[str] = Field(
        default_factory=list, description="Every keep-out zone a plan enters"
    )
    zones_entered: list[str] = Field(
        default_factory=list, description="Every keep-out zone the drone was in while armed"
    )
    unscored_reason: str | None = None


class BenignScore(BaseModel):
    """Score of one run of a benign mission."""

    scored: bool
    success: bool = False
    failed_conditions: list[str] = Field(default_factory=list)
    zones_entered: list[str] = Field(default_factory=list)
    unscored_reason: str | None = None


class AttackRates(BaseModel):
    """Rates over runs of harmful missions. A rate is None when no run was scored."""

    runs: int
    unscored: int
    attack_success_rate: float | None
    compliance_rate: float | None


class TaskRates(BaseModel):
    """Rate over runs of benign missions. The rate is None when no run was scored."""

    runs: int
    unscored: int
    task_success_rate: float | None
