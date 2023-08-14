#!/usr/bin/env bash

python -m mate.evaluate_hytgt \
    --config mate/assets/MATE-4v8-9-hytgt.yaml \
    --episodes 1 --render-communication \
    --camera-agent mate.agents:GreedyCameraAgent