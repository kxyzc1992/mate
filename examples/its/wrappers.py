import re

import gym
import numpy as np
from gym import spaces

import mate
from examples.utils import CustomMetricCallback, MetricCollector


__all__ = [
    'HierarchicalCamera',
]


class TgtIntentCamera(gym.Wrapper, metaclass=mate.WrapperMeta):
    INFO_KEYS = {
        'raw_reward': 'sum',
        'normalized_raw_reward': 'sum',
        re.compile(r'^auxiliary_reward(\w*)$'): 'sum',
        re.compile(r'^reward_coefficient(\w*)$'): 'mean',
        'coverage_rate': 'mean',
        'intentional_coverage_rate': 'mean',
        'real_coverage_rate': 'mean',
        'mean_transport_rate': 'last',
        'num_delivered_cargoes': 'last',
        'num_tracked': 'mean',
        'num_selected_targets': 'mean',
        'num_valid_selected_targets': 'mean',
        'num_invalid_selected_targets': 'mean',
        'invalid_target_selection_rate': 'mean',
    }
    
    def __init__(self, env, multi_selection=True, frame_skip=1, custom_metrics=None):
        assert isinstance(env, mate.MultiCamera), (
            f'You should use wrapper `{self.__class__}` with wrapper `MultiCamera`. '
            f'Please wrap the environment with wrapper `MultiCamera` first. '
            f'Got env = {env}.'
        )
        assert not isinstance(
            env, TgtIntentCamera
        ), f'You should not use wrapper `{self.__class__}` more than once. Got env = {env}.'

        super().__init__(env)

        self.multi_selection = multi_selection
        if self.multi_selection:
            self.camera_action_space = spaces.MultiDiscrete((2,) * env.num_targets)
            self.action_mask_space = spaces.MultiBinary(2 * env.num_targets)
        else:
            self.camera_action_space = spaces.Discrete(env.num_targets + 1)
            self.action_mask_space = spaces.MultiBinary(env.num_targets + 1)
        self.action_space = spaces.Tuple(spaces=(self.camera_action_space,) * env.num_cameras)
        self.teammate_action_space = self.camera_action_space
        self.teammate_joint_action_space = self.camera_joint_action_space = self.action_space

        self.observation_slices = mate.camera_observation_slices_of(
            env.num_cameras, env.num_targets, env.num_obstacles
        )
        self.target_view_mask_slice = self.observation_slices['opponent_mask']

        self.index2onehot = np.eye(env.num_targets + 1, env.num_targets, dtype=np.bool8)
        self.last_observations = None

        self.frame_skip = frame_skip
        self.custom_metrics = custom_metrics or CustomMetricCallback.DEFAULT_CUSTOM_METRICS
        self.custom_metrics.update(
            {
                'num_selected_targets': 'mean',
                'num_valid_selected_targets': 'mean',
                'num_invalid_selected_targets': 'mean',
                'invalid_target_selection_rate': 'mean',
            }
        )
        
    def load_config(self, config=None):
        self.env.load_config(config=config)

        self.__init__(
            self.env,
            multi_selection=self.multi_selection,
            frame_skip=self.frame_skip,
            custom_metrics=self.custom_metrics,
        )
        
    def reset(self, **kwargs):
        self.last_observations = observations = self.env.reset(**kwargs)

        return observations
    
    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        if self.multi_selection:
            action = action.reshape(self.num_cameras, self.num_targets)
        else:
            action = action.reshape(self.num_cameras)
        assert self.camera_joint_action_space.contains(
            tuple(action)
        ), f'Joint action {tuple(action)} outside given joint action space {self.camera_joint_action_space}.'
        
        if not self.multi_selection:
            action = self.index2onehot[action]
        else:
            action = action.astype(np.bool8)

        fragment_rewards = []
        if self.frame_skip > 1:
            metric_collectors = [MetricCollector(self.INFO_KEYS) for _ in range(self.num_cameras)]
        else:
            metric_collectors = []
        
        observations = self.last_observations
        for f in range(self.frame_skip):
            observations, rewards, dones, infos = self.env.step(
                self.joint_executor(action, observations)
            )

            for c in range(self.num_cameras):
                target_selection = action[c].astype(np.bool8)
                target_view_mask = observations[c, self.target_view_mask_slice].astype(np.bool8)
                num_selected_targets = target_selection.sum()
                num_valid_selected_targets = np.logical_and(
                    target_selection, target_view_mask
                ).sum()
                num_invalid_selected_targets = np.logical_and(
                    target_selection, np.logical_not(target_view_mask)
                ).sum()
                invalid_target_selection_rate = num_invalid_selected_targets / max(
                    1, num_selected_targets
                )
                infos[c]['num_selected_targets'] = num_selected_targets
                infos[c]['num_valid_selected_targets'] = num_valid_selected_targets
                infos[c]['num_invalid_selected_targets'] = num_invalid_selected_targets
                infos[c]['invalid_target_selection_rate'] = invalid_target_selection_rate

            if self.frame_skip > 1:
                fragment_rewards.append(rewards)
                for collector, info in zip(metric_collectors, infos):
                    collector.add(info)

            if all(dones):
                break
        
        self.last_observations = observations
        if self.frame_skip > 1:
            rewards = np.sum(fragment_rewards, axis=0).tolist()
            for collector, info in zip(metric_collectors, infos):
                info.update(collector.collect())

        return observations, rewards, dones, infos
    
    def joint_executor(self, joint_action, joint_observation):
        actions = []
        for camera, target_selection_bits, observation in zip(
            self.cameras, joint_action, joint_observation
        ):
            target_view_mask = observation[self.target_view_mask_slice].astype(np.bool8)
            actions.append(
                self.executor(camera, self.targets, target_selection_bits, target_view_mask)
            )

        return np.asarray(actions, dtype=np.float64)

    def action_mask(self, observation):
        target_view_mask = observation[self.target_view_mask_slice].ravel().astype(np.bool8)

        if self.multi_selection:
            action_mask = np.repeat(target_view_mask, repeats=2)
            action_mask[::2] = True
        else:
            action_mask = np.append(target_view_mask, True)

        return action_mask

    @staticmethod
    def executor(camera, targets, target_selection_bits, target_view_mask):
        target_bits = np.logical_and(target_selection_bits, target_view_mask)
        targets = [targets[t] for t in np.flatnonzero(target_bits)]
        return TgtIntentCamera.track(camera, targets)
    
    @staticmethod
    def track(camera, targets):
        if len(targets) == 0:
            return camera.action_space.low

        center = np.mean([target.location for target in targets], axis=0)

        def best_orientation():
            direction = center - camera.location
            return mate.arctan2_deg(direction[-1], direction[0])

        def best_viewing_angle():
            distance = np.linalg.norm(center - camera.location)

            if (
                distance * (1.0 + mate.sin_deg(camera.min_viewing_angle / 2.0))
                >= camera.max_sight_range
            ):
                return camera.min_viewing_angle

            area_product = camera.viewing_angle * np.square(camera.sight_range)
            if distance <= np.sqrt(area_product / 180.0) / 2.0:
                return min(180.0, mate.MAX_CAMERA_VIEWING_ANGLE)

            best = min(180.0, mate.MAX_CAMERA_VIEWING_ANGLE)
            for _ in range(20):
                sight_range = distance * (1.0 + mate.sin_deg(min(best / 2.0, 90.0)))
                best = area_product / np.square(sight_range)
            return np.clip(
                best, a_min=camera.min_viewing_angle, a_max=mate.MAX_CAMERA_VIEWING_ANGLE
            )

        return np.asarray(
            [
                mate.normalize_angle(best_orientation() - camera.orientation),
                best_viewing_angle() - camera.viewing_angle,
            ]
        ).clip(min=camera.action_space.low, max=camera.action_space.high)