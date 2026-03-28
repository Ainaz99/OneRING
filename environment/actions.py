import json
from typing import Optional, Dict, Union, Any, Type, List


class AbstractAction:
    def __init__(
        self,
        short_name: Optional[str] = None,
        long_name: Optional[str] = None,
    ):
        self.short_name = short_name
        self.long_name = long_name
        self.unit_info = {}

    def enact(self):
        raise NotImplementedError

    def to_string(self) -> str:
        raise NotImplementedError


class StretchAction(AbstractAction):
    ramping_adjustment = 0.707  # RMS as a rough approximation from max speed
    max_speeds = {
        "base_z": ramping_adjustment * 0.3,  # meters per second
        "base_theta": ramping_adjustment * 45,  # degrees per second
        "arm_y": ramping_adjustment * 0.15,  # meters per second
        "arm_z": ramping_adjustment * 0.30,  # meters per second
        "wrist_yaw": ramping_adjustment * 45,  # degrees per second
    }

    def __init__(
        self,
        base: Optional[Dict[str, Union[int, float]]] = None,
        arm: Optional[Dict[str, Union[int, float]]] = None,
        wrist: Optional[Dict[str, Union[int, float]]] = None,
        gripper: Optional[Dict[str, Union[int, float]]] = None,
        absolute_arm_and_wrist_motion: bool = False,
        long_name: Optional[str] = None,
        short_name: Optional[str] = None,
    ):
        super().__init__(long_name=long_name, short_name=short_name)
        base_defaults = {"z": None, "theta": None}
        arm_defaults = {"y": None, "z": None}
        wrist_defaults = {"yaw": None}
        gripper_defaults = {"gripper_openness": None}

        self._base = {**base_defaults, **(base or {})}
        self._arm = {**arm_defaults, **(arm or {})}
        self._wrist = {**wrist_defaults, **(wrist or {})}
        self._gripper = {**gripper_defaults, **(gripper or {})}
        self.absolute_arm_and_wrist_motion = absolute_arm_and_wrist_motion

        self.unit_info = {
            "base": {
                "z": "meters",
                "theta": "degrees",
            },
            "arm": {
                "y": "meters",
                "z": "meters",
            },
            "wrist": {
                "yaw": "degrees",
            },
            "gripper": {
                "gripper_openness": "percentage",
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        def filter_none_values(d):
            if isinstance(d, dict):
                return {k: filter_none_values(v) for k, v in d.items() if v is not None}
            else:
                return d

        full_dict = {
            "base": self._base,
            "arm": self._arm,
            "wrist": self._wrist,
            "gripper": self._gripper,
        }
        final_filter = {k: v for k, v in filter_none_values(full_dict).items() if v}
        return final_filter

    def to_string(self) -> str:
        if self.long_name:
            return self.long_name
        return json.dumps(self.to_dict())

    def __str__(self) -> str:
        return self.to_string()

    def time_estimate(self) -> float:
        # Base motions happen sequentially
        base_time = (
            abs(self._base["z"] or 0) / self.max_speeds["base_z"]
            + abs(self._base["theta"] or 0) / self.max_speeds["base_theta"]
        )

        # Arm, wrist, and gripper motions can happen independently
        arm_wrist_gripper_time = max(
            abs(self._arm["y"] or 0) / self.max_speeds["arm_y"],
            abs(self._arm["z"] or 0) / self.max_speeds["arm_z"],
            abs(self._wrist["yaw"] or 0) / self.max_speeds["wrist_yaw"],
            (
                1.0 if self._gripper["gripper_openness"] is not None else 0.0
            ),  # Assuming 1 second for gripper action
        )

        # Return the maximum of base time and arm/wrist/gripper time
        return max(base_time, arm_wrist_gripper_time)

    @classmethod
    def init_from_singletons(cls, singleton_actions: List["StretchAction"]) -> "StretchAction":
        # Initialize empty dictionaries for each component
        base = {"z": None, "theta": None}
        arm = {"y": None, "z": None}
        wrist = {"yaw": None}
        gripper = {"gripper_openness": None}

        for action in singleton_actions:
            if action._base["z"] is not None:
                assert base["z"] is None, "Duplicate base z action"
                base["z"] = action._base["z"]
            if action._base["theta"] is not None:
                assert base["theta"] is None, "Duplicate base theta action"
                base["theta"] = action._base["theta"]
            if action._arm["y"] is not None:
                assert arm["y"] is None, "Duplicate arm y action"
                arm["y"] = action._arm["y"]
            if action._arm["z"] is not None:
                assert arm["z"] is None, "Duplicate arm z action"
                arm["z"] = action._arm["z"]
            if action._wrist["yaw"] is not None:
                assert wrist["yaw"] is None, "Duplicate wrist yaw action"
                wrist["yaw"] = action._wrist["yaw"]
            if action._gripper["gripper_openness"] is not None:
                assert gripper["gripper_openness"] is None, "Duplicate gripper action"
                gripper["gripper_openness"] = action._gripper["gripper_openness"]

        # Create a new instance of the StretchAction class with the combined actions
        return cls(base=base, arm=arm, wrist=wrist, gripper=gripper)

    def get_scalar_value(self) -> float:
        """
        Get the scalar value of a singleton StretchAction.

        :return: The scalar value of the action.
        :raises AssertionError: If the action is not a singleton.
        :raises ValueError: If no scalar value is found in the action.
        """

        assert len(self.enact()) <= 1 and not (
            self._arm["y"] is not None and self._arm["z"] is not None
        ), "This method is only for singleton actions."

        if self._base["z"] is not None:
            return self._base["z"]
        elif self._base["theta"] is not None:
            return self._base["theta"]
        elif self._arm["y"] is not None:
            return self._arm["y"]
        elif self._arm["z"] is not None:
            return self._arm["z"]
        elif self._wrist["yaw"] is not None:
            return self._wrist["yaw"]
        elif self._gripper["gripper_openness"] is not None:
            return self._gripper["gripper_openness"]

        # If no scalar value found
        raise ValueError("No scalar value found in this StretchAction")

    def enact(self):
        if self.absolute_arm_and_wrist_motion:
            raise NotImplementedError("Absolute arm and wrist motion not yet implemented.")

        flattened_action_components = []
        if self._base["theta"] is not None:
            flattened_action_components.append(
                {"action": "RotateAgent", "degrees": self._base["theta"]}
            )
        if self._base["z"] is not None:
            flattened_action_components.append({"action": "MoveAgent", "ahead": self._base["z"]})
        if self._arm["y"] or self._arm["z"]:
            flattened_action_components.append(
                {
                    "action": "MoveArmRelative",
                    "offset": {"x": 0, "y": self._arm["y"] or 0, "z": self._arm["z"] or 0},
                }
            )
        if self._wrist["yaw"]:
            flattened_action_components.append(
                {"action": "RotateWristRelative", "yaw": self._wrist["yaw"]}
            )
        if self._gripper["gripper_openness"]:
            raise NotImplementedError("Gripper openness not yet implemented.")
            # flattened_action_components.append(
            #     {"action": "SetGripper", "gripper_openness": self._gripper["gripper_openness"]}
            # )
        return flattened_action_components


class StretchGraspAction(StretchAction):
    def enact(self):
        return [{"action": "PickupObject"}]


class StretchDropOffAction(StretchAction):
    def enact(self):
        return [{"action": "ReleaseObject"}]


class StretchActionReal(StretchAction):
    def enact(self):
        if self.absolute_arm_and_wrist_motion:
            raise NotImplementedError("Absolute arm and wrist motion not yet implemented.")

        action_args = {}

        if self._base["z"] is not None:
            action_args["base_z"] = self._base["z"]
        if self._base["theta"] is not None:
            action_args["base_theta"] = self._base["theta"]
        if self._arm["y"] is not None:
            action_args["arm_y"] = self._arm["y"]
        if self._arm["z"] is not None:
            action_args["arm_z"] = self._arm["z"]
        if self._wrist["yaw"] is not None:
            action_args["wrist_yaw"] = self._wrist["yaw"]
        if self._gripper["gripper_openness"] is not None:
            action_args["gripper_openness"] = self._gripper["gripper_openness"]

        if action_args:
            return [{"action": "FullyParameterizedSpatialAction", "args": action_args}]
        else:
            return []


class StretchGraspActionReal:
    # TODO
    pass


class StretchDropOffActionReal:
    # TODO
    pass
