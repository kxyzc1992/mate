"""Example of MAPPO agents for the Multi-Agent Tracking Environment."""

from examples.mappo import camera, target
from examples.mappo_hytgt.camera import MAPPOCameraAgent
from examples.mappo_hytgt.target import MAPPOTargetAgent


CameraAgent = MAPPOCameraAgent
TargetAgent = MAPPOTargetAgent
