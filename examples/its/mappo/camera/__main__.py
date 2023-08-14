#!/usr/bin/env python3

r"""
Example of TargetIntent MAPPO agents for the Multi-Agent Tracking Environment.

.. code:: bash

    python3 -m examples.its.mappo.camera

    python3 -m mate.evaluate --episodes 1 --render-communication \
        --camera-agent examples.its:ITSMAPPOCameraAgent \
        --camera-kwargs '{ "checkpoint_path": "examples/its/mappo/camera/ray_results/ITS-MAPPO/latest-checkpoint" }'
"""

import argparse
import functools
import os
import sys

import mate
from examples.its.mappo.camera.agent import ITSMAPPOCameraAgent
from examples.its.mappo.camera.train import experiment
from examples.its.wrappers import TgtIntentCamera


CHECKPOINT_PATH = os.path.join(experiment.checkpoint_dir, 'latest-checkpoint')

MAX_EPISODE_STEPS = 4000


def main():
    parser = argparse.ArgumentParser(prog=f'python -m {__package__}')
    parser.add_argument(
        '--checkpoint-path',
        '--checkpoint',
        '--ckpt',
        type=str,
        metavar='PATH',
        default=CHECKPOINT_PATH,
        help='path to the checkpoint file',
    )
    parser.add_argument(
        '--max-episode-steps',
        type=int,
        metavar='STEP',
        default=MAX_EPISODE_STEPS,
        help='maximum episode steps (default: %(default)d)',
    )
    parser.add_argument(
        '--seed', type=int, metavar='SEED', default=0, help='the global seed (default: %(default)d)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(
            (
                f'Model checkpoint ("{args.checkpoint_path}") does not exist. Please run the following command to train a model first:\n'
                f'  python -m examples.its.mappo.camera.train'
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    # Make agents ##############################################################
    camera_agent = ITSMAPPOCameraAgent(checkpoint_path=args.checkpoint_path)
    target_agent = mate.GreedyTargetAgent()

    # Make the environment #####################################################
    env_config = camera_agent.config.get('env_config', {})
    enhanced_observation_team = str(env_config.get('enhanced_observation', None)).lower()

    base_env = mate.make(
        'MultiAgentTracking-v0',
        config=env_config.get('config'),
        **env_config.get('config_overrides', {}),
    )
    base_env = mate.RenderCommunication(base_env)
    if enhanced_observation_team is not None:
        base_env = mate.EnhancedObservation(base_env, team=enhanced_observation_team)
    env = mate.MultiCamera(base_env, target_agent=target_agent)
    print(env)

    # Rollout ##################################################################
    camera_agents = camera_agent.spawn(env.num_cameras)

    camera_joint_observation = env.reset()
    env.render()

    mate.group_reset(camera_agents, camera_joint_observation)
    camera_infos = None

    for i in range(MAX_EPISODE_STEPS):
        camera_joint_action = mate.group_step(
            env, camera_agents, camera_joint_observation, camera_infos
        )

        selections = [
            (agent.index, agent.last_selection, agent.last_mask) for agent in camera_agents
        ]

        results = env.step(camera_joint_action)
        camera_joint_observation, camera_team_reward, done, camera_infos = results

        render_callback = functools.partial(
            TgtIntentCamera.render_selection_callback, selections=selections
        )
        env.render(onetime_callbacks=[render_callback])
        if done:
            break


if __name__ == '__main__':
    main()
