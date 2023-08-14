#!/usr/bin/env bash

python -m mate.evaluate_hytgt \
    --config mate/assets/MATE-4v8-9-hytgt.yaml \
    --episodes 1 --render-communication \
    # --camera-agent examples.mappo_hytgt:MAPPOCameraAgent \
    --camera-agent examples.mappo:MAPPOCameraAgent \
    --camera-kwargs '{ "checkpoint_path": "examples/mappo_hytgt/camera/ray_results/MAPPO/latest-checkpoint" }'