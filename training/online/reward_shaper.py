from typing import List

from allenact.utils.misc_utils import prepare_locals_for_super

from tasks.abstract_task import AbstractVIDATask


class RewardShaper:
    def __init__(self, task: AbstractVIDATask) -> None:
        self.task = task
        self.task_info = task.task_info
        self.reward_config = task.reward_config
        self.action_names = task.action_names
        self.controller = task.controller

        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []

        self.distance_type = None
        self.dist_to_target_func = None

    def shaping(self) -> float:
        raise NotImplementedError


class ObjectNavRewardShaper(RewardShaper):
    def __init__(
        self,
        task: AbstractVIDATask,
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))
        self.distance_type = self.task.distance_type
        self.dist_to_target_func = self.task.dist_to_target_func
        last_distance = self.dist_to_target_func()
        self.closest_distance = float(last_distance)

    def shaping(self) -> float:
        if self.reward_config is None:
            return 0
        if self.reward_config.shaping_weight == 0.0:
            return 0

        reward = 0.0
        cur_distance = self.dist_to_target_func()

        if not self.task.last_action_success:
            if not self.task._took_end_action:
                reward += self.reward_config.failed_action_penalty

        if self.distance_type == "l2":
            reward += self.reward_config.shaping_weight * max(
                self.closest_distance - cur_distance, 0
            )
            self.closest_distance = min(self.closest_distance, cur_distance)

        return reward


