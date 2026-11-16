"""Red-teaming benchmark for LLM-planned drones: missions, attacks and deterministic scoring.

The benchmark is data (``world.json``, ``missions.json``) plus pure functions that
score one run from its plan and its flight log. Nothing in this package needs ROS.

Run the tests from the repository root::

    pytest tests/redteam
"""
