import inspect
import os
import random
import time
import traceback
import warnings
from contextlib import contextmanager
from typing import Dict, Optional, Set, Sequence, List, Tuple, Iterable, Literal, Union
import sys

import numpy as np
import torch
from ai2thor.controller import Controller
from ai2thor.server import Event
from shapely import Polygon, GeometryCollection, Point

from environment.action_spaces import agent_alignment_to_point
from environment.actions import StretchAction, StretchGraspAction, StretchDropOffAction
from environment.stretch_state import StretchState
from environment.vida_objects import VIDAObject
from utils.constants.objaverse_data_dirs import OBJAVERSE_ASSETS_VERSION
from utils.constants.stretch_initialization_utils import (
    AGENTS_BASE_HEIGHT,
    AGENT_RADIUS_LIST,
    GRID_SIZE,
    ADDITIONAL_ARM_ARGS,
    HORIZON,
    ADDITIONAL_NAVIGATION_ARGS,
    STRETCH_COMMIT_ID,
    INTEL_VERTICAL_FOV,
)
from utils.data_generation_utils.navigation_utils import (
    get_rooms_polymap_and_type,
    get_room_id_from_location,
    get_wall_center_floor_level,
    triangulate_room_polygon,
    is_any_object_sufficiently_visible_and_in_center_frame,
    snap_to_skeleton,
)
from utils.distance_calculation_utils import position_dist
from utils.distance_calculation_utils import sum_dist_path
from utils.data_generation_utils.camera_utils import (
    calc_camera_intrinsics,
)
from utils.synsets.hypernyms import is_hypernym_of
from utils.type_utils import Vector3


def generate_unique_color(object_id):
    random.seed(object_id)
    return [random.randint(0, 255) for _ in range(3)]


def color_instance_frame(instance_frame):
    unique_ids = np.unique(instance_frame)
    color_map = {obj_id: generate_unique_color(obj_id) for obj_id in unique_ids}
    colored_frame = np.zeros((*instance_frame.shape, 3), dtype=np.uint8)
    for obj_id, color in color_map.items():
        colored_frame[instance_frame == obj_id] = color
    return colored_frame

def rgb2uint32(r, g, b, alpha=255):
    return np.uint32(((alpha & 255) << 24) | ((b & 255) << 16) | ((g & 255) << 8) + r)


class LazyCamera:
    # centralized location for everything relating to a specific camera,
    # which is a) lazy and b) pretending to be a dict.
    # I find this useful and intuitive, ping me if you hate it -RMH

    # sample access pattern inside the controller class:
    # self.camera_registry["nav"]["rgb"]
    # self.camera_registry["manip"]["camera_parameters"]
    def __init__(self, controller, index, name, thor_name="unknown"):
        self.controller = controller
        self.index = index
        self.name = name
        self.thor_name = thor_name
        self._cache = {}
        self._pose = dict(position=None, rotation=None)

        self.fixed_properties = {
            "index": self.index,
            "name": self.name,
            "thor_name": self.thor_name,
        }

        self.lazy_properties = {
            "segmentation": lambda: self.controller.segmentation_frame(self.name),
            "visible_objects_2m": lambda: self.controller.get_visible_objects(self.name),
            "rgb": lambda: self.controller.rgb_frame(self.name),
            "camera_parameters": lambda: self.controller.get_camera_parameters(self.name),
            "commanded_pose": lambda: self._pose,
        }

    def _lazy_load(self, attr, func):
        if attr not in self._cache:
            self._cache[attr] = func()
        return self._cache[attr]

    def __getitem__(self, key):
        if key in self.fixed_properties:
            return self.fixed_properties[key]
        elif key in self.lazy_properties:
            return self._lazy_load(key, self.lazy_properties[key])
        else:
            raise KeyError(f"Unknown key: {key}")

    def stepwise_reset(self):
        non_reset_properties = [
            "camera_parameters",
        ]
        self._cache = {
            key: value for key, value in self._cache.items() if key in non_reset_properties
        }

    def task_reset(self, randomize_distortion_parameters=False, preset_distortion_parameters=None):
        # Parameters kept for compatibility but not used (no warping)
        _ = randomize_distortion_parameters
        _ = preset_distortion_parameters
        self._cache = {}

    def reset_commanded_position_and_rotation(self, position, rotation):
        self._pose["position"] = position
        self._pose["rotation"] = rotation

    def keys(self):
        return self.fixed_properties.keys() | self.lazy_properties.keys()

    def __contains__(self, key):
        return key in self.keys()

    def __iter__(self):
        return iter(self.fixed_properties | self.lazy_properties)


class LazyCameraRegistry:
    def __init__(self, controller, camera_type=LazyCamera):
        self.controller = controller
        self.cameras = {
            "nav": camera_type(controller, name="nav", index="main", thor_name="FirstPersonCharacter"),
            "manip": camera_type(controller, name="manip", index=0, thor_name="SecondaryCamera"),
        }

    def __getitem__(self, key):
        try:
            return self.cameras[key]
        except KeyError:
            # get the full trace, who is trying to call a camera that's not available?
            print(f"Invalid camera key {key} is being called in this function stack: ")
            print("\n".join(frame.function for frame in inspect.stack()))
            raise

    def __getattr__(self, key):
        if key in self.cameras:
            return self.cameras[key]
        raise AttributeError(f"'LazyCameraRegistry' object has no attribute '{key}'")

    def stepwise_reset(self):
        # reset relevant visibility caches between steps, but not camera properties
        for camera in self.cameras.values():
            camera.stepwise_reset()

    def task_reset(self, randomize_distortion_parameters=False, preset_distortion_parameters=None):
        # reset all values including camera properties
        for camera in self.cameras.values():
            camera.task_reset(
                randomize_distortion_parameters,
                preset_distortion_parameters,
            )

    def keys(self):
        return self.cameras.keys()

    def __contains__(self, key):
        return key in self.cameras

    def __iter__(self):
        return iter(self.cameras)



class OriginalStretchController:
    def __init__(self, initialize_controller=True, **kwargs):
        self.should_render_image_synthesis = (
            kwargs.get("renderDepthImage", False)
            or kwargs.get("renderNormalsImage", False)
            or kwargs.get("renderFlowImage", False)
            or kwargs.get("renderDistortionImage", False)
        )
        self.mode = None

        self.room_poly_map: Optional[Dict[str, Polygon]] = None
        self.room_type_dict: Optional[Dict[str, str]] = None

        # The usage here is to judge if a spatial action counts as successful or not
        self._universal_state_tolerance = StretchState._create_difference_state(
            diff_base={"x": 0.01, "z": 0.01, "theta": 1.5},
            diff_wrist={"y": 0.005, "z": 0.005, "yaw": 2},
            diff_hand={
                "x": 100,
                "y": 100,
                "z": 100,
            },  # direct hand is a no-op
            diff_gripper=100,
            diff_held_oids=set(),
        )

        self._current_horizon = 0
        self._visible_objects_cache = {
            "nav": {},
            "manip": {},
        }
        self.camera_registry = LazyCameraRegistry(self)

        if initialize_controller:
            pid = os.getpid()
            print(f"Initializing Controller with PID: {pid}")
            unityLogFilePath = (
                f"/data/input/datasets/vida_datasets/unity_logs/process_{pid}_{int(time.time())}_unity.log"
                if sys.platform == "linux"
                else os.path.expanduser(
                    f"~/Desktop/holodeck_vida/unity_logs/process_{pid}_{int(time.time())}_unity.log"
                )
            )
            kwargs["unityLogFilePath"] = unityLogFilePath
            self.controller = Controller(**kwargs)
            self.initialization_args = kwargs
            print(
                f"PID {pid} using Controller commit id: {self.controller._build.commit_id}"
                f" with Unity log file path: {unityLogFilePath}"
            )
            assert STRETCH_COMMIT_ID in self.controller._build.commit_id

            if "scene" in kwargs:
                self.reset(kwargs["scene"])

            self.reset_agent_embodiment(initialize_cameras=True, randomize_embodiment=True)
            assert (
                abs(self.camera_registry["nav"]["camera_parameters"]["fov"] - INTEL_VERTICAL_FOV)
                < 5
            ), (
                f"The nav camera's vertical FOV should be close to {INTEL_VERTICAL_FOV} degrees."
                f" Instead it is {self.camera_registry['nav']['camera_parameters']['fov']}"
            )

    def put_object_in_hand(self, object_id: str):
        self.step(
            action="RotateWristRelative",
            yaw=-180,
        )
        self.step(action="MoveArmRelative", offset={"x": 0, "y": 0.4, "z": 0.05})
        self.step(
            action="SetGripperOpenness",
            openness=50,
            raise_for_failure=True,
        )
        return self.step(action="PlaceObjectIntoGripper", objectId=object_id)

    # TODO RMH THESE FUNCTIONS SHOULD USE STRETCHSTATE
    def get_arm_sphere_center(self):
        return self.controller.last_event.metadata["arm"]["handSphereCenter"]

    def get_wrist_center(self):
        wrist_center = self.controller.last_event.metadata["arm"]["joints"][-2]
        assert wrist_center["name"] == "stretch_robot_wrist_1_jnt"
        return wrist_center["position"]

    def dist_from_arm_to_obj(self, object_id):
        object_location = [self.get_object_position(object_id)[k] for k in ["x", "y", "z"]]
        arm_location = self.get_arm_wrist_absolute_position()
        return (torch.Tensor(arm_location) - torch.Tensor(object_location)).norm().item()

    def dist_from_arm_sphere_center_to_obj(self, object_id):
        return position_dist(
            self.get_object_position(object_id), self.get_arm_sphere_center(), ignore_y=False
        )

    def dist_from_arm_sphere_center_to_obj_colliders_closest_to_point(self, object_id):
        arm_sphere_center = self.get_arm_sphere_center()
        points_on_obj = self.controller.step(
            action="PointOnObjectsCollidersClosestToPoint",
            objectId=object_id,
            point=arm_sphere_center,
        ).metadata["actionReturn"]
        if points_on_obj is None or len(points_on_obj) == 0:
            return self.dist_from_arm_sphere_center_to_obj(object_id)
        else:
            dists = [position_dist(arm_sphere_center, p, ignore_y=False) for p in points_on_obj]
        return min(dists)

    def get_floor_level(self):
        return self.controller.last_event.metadata["agent"]["position"]["y"] - AGENTS_BASE_HEIGHT

    @property
    def navigation_camera(self):
        """Compatibility property for sensors that expect navigation_camera."""
        frame = self.camera_registry["nav"]["rgb"]
        # Apply the same cutoff as in the original implementation
        cutoff = round(frame.shape[1] * 6 / 396)
        return frame[:, cutoff:-cutoff, :]

    @property
    def manipulation_camera(self):
        """Compatibility property for sensors that expect manipulation_camera."""
        frame = self.camera_registry["manip"]["rgb"]
        # Apply the same cutoff as in the original implementation
        cutoff = round(frame.shape[1] * 6 / 396)
        return frame[:, cutoff:-cutoff, :3]

    def segmentation_frame(
        self, which_camera: Literal["nav", "manip"] = "nav"
    ):
        called_for_object = "get_segmentation_mask_of_object" in [
            f.function for f in inspect.stack()
        ]
        called_for_pixel_ref = "get_object_from_pixel" in [f.function for f in inspect.stack()]
        assert called_for_object or called_for_pixel_ref, (
            f"Segmentation frame should only be called via get_segmentation_mask_of_object - called in {inspect.stack()}"
        )
        # IMPORTANT NOTE: this assert may have a de facto bypass with how cacheing works.
        # TODO: it would be good to have a more robust way to ensure this is masked without ruining the laziness.
        # the empty_mask property seems promisingly named but is not writeable ?or seemingly used?.
        if self.controller.last_event.instance_segmentation_frame is None:
            self.controller.step("Pass", renderImageSynthesis=True)
            assert self.controller.last_event.instance_segmentation_frame is not None, (
                "Must pass `renderInstanceSegmentation=True` on initialization"
                " to obtain a camera_segmentation"
            )
        if self.camera_registry[which_camera]["index"] == "main":
            return self.controller.last_event.instance_masks
        else:
            return self.controller.last_event.third_party_instance_masks[
                self.camera_registry[which_camera]["index"]
            ]

    def rgb_frame(self, which_camera: Literal["nav", "manip"]):
        if self.camera_registry[which_camera]["index"] == "main":
            return self.controller.last_event.frame[:, :, :3]
        camera_index = self.camera_registry[which_camera]["index"]
        if camera_index is None or camera_index >= len(
            self.controller.last_event.third_party_camera_frames
        ):
            # Get the shape of the nav camera frame
            nav_frame_shape = self.camera_registry["nav"]["rgb"].shape
            return np.zeros(
                nav_frame_shape, dtype=np.uint8
            )  # Return a blank frame with correct size
        return self.controller.last_event.third_party_camera_frames[camera_index][:, :, :3]

    @staticmethod
    def frame_resize(frame, target_height=None, target_width=None):
        # deprecated
        raise NotImplementedError(
            "frame_resize is deprecated because CPU resizes are slow - resize separately for eval or have it in augmentation for train."
        )
        # if the frame is already the right size, just return it
        if frame.shape[0] == target_height and frame.shape[1] == target_width:
            return frame
        else:
            raise ValueError(
                f"Frame shape {frame.shape} does not match target shape {target_height}x{target_width}"
            )
        # # this is to handle pixelation in the warped frame when getting it from unity
        # # assert that the frame shape and target shape wont change the aspect ratio
        # assert frame.shape[0] / frame.shape[1] == target_height / target_width
        # big_distorted_tensor = (
        #     torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        # )

        # # Efficient resizing with torch interpolate
        # resized_tensor = torch.nn.functional.interpolate(
        #     big_distorted_tensor,
        #     size=(target_height, target_width),
        #     mode="bilinear",
        #     align_corners=False,
        # )

        # # Convert back to numpy array if needed and remove batch dimension
        # resized_frame = (resized_tensor.squeeze(0).permute(1, 2, 0) * 255).byte().numpy()
        # return resized_frame


    def get_segmentation_mask_of_object(
        self,
        object_id: str,
        which_camera: Literal["nav", "manip"],
    ):
        segmentation_to_look_at = self.camera_registry[which_camera]["segmentation"]
        # save this frame as a test
        mask: np.ndarray[bool] | None = None

        if object_id in segmentation_to_look_at:
            mask = segmentation_to_look_at[object_id]
        elif object_id in segmentation_to_look_at.class_colors:
            best_count = 0
            for color in segmentation_to_look_at.class_colors[object_id]:
                current_mask = (
                    segmentation_to_look_at.instance_segmentation_frame_uint32 == rgb2uint32(*color)
                )
                current_count = current_mask.sum()
                if current_count > best_count:
                    best_count = current_count
                    mask = current_mask

        if mask is not None:
            if np.sum(mask) > 0:
                return mask

        return np.zeros(self.camera_registry["nav"]["rgb"].shape[:2], dtype=bool)

    def get_pixel_mass_of_object(
        self, object_id: str, which_camera: str, return_largest_mask=False
    ):
        # light wrapper around get_segmentation_mask_of_object function to account for "both" cameras. should sum in that case
        if which_camera == "both":
            cameras_to_check = ["nav", "manip"]
        else:
            cameras_to_check = [which_camera]

        total_pixel_mass = 0
        largest_mass = 0
        largest_mask = None
        largest_camera = None

        for camera in cameras_to_check:
            mask = self.get_segmentation_mask_of_object(object_id, camera)
            mask_sum = np.sum(mask)
            total_pixel_mass += mask_sum

            if mask_sum > largest_mass:
                largest_mass = mask_sum
                largest_mask = mask
                largest_camera = camera

        if return_largest_mask:
            return total_pixel_mass, (largest_camera, largest_mask)
        return total_pixel_mass

    def step(self, **kwargs):
        if "renderImageSynthesis" not in kwargs:
            kwargs["renderImageSynthesis"] = self.should_render_image_synthesis

        if kwargs["action"] in ["Teleport", "TeleportFull"]:
            # We don't want users to call teleport directly because this can mess up the camera horizon
            raise NotImplementedError(
                f"Use `teleport_agent` instead of `step` for teleportation (attempted action: {kwargs['action']})."
            )

        if kwargs["action"] == "__Teleport__":
            # This is how we allow the stretch agent itself to call Teleport itself without raising an error
            kwargs["action"] = "TeleportFull"

        return self.controller.step(**kwargs)

    def teleport_agent(self, position: Vector3, rotation: Union[Vector3, float], **kwargs) -> Event:
        if isinstance(rotation, Dict):
            rotation = rotation["y"]

        # if "standing" in kwargs:
        kwargs["standing"] = None

        # if "horizon" in kwargs:
        kwargs["horizon"] = None
        # warnings.warn(
        #     "`horizon` is not a valid argument for teleport_agent, as camera locations are set on reset."
        #     " This argument will be ignored."
        # )

        # if len(kwargs) > 0:
        #     allowed_keys = {
        #         "forceAction",
        #         "renderImage",
        #         "renderImageSynthesis",
        #         "raise_for_failure",
        #         "agentId",
        #     }
        #     assert set(kwargs.keys()).issubset(
        #         allowed_keys
        #     ), f"Invalid arguments for teleport_agent: {set(kwargs.keys()) - allowed_keys}"

        # RMH NOTE: THESE MUST BE BYPASSED, FOR THE MADNESS. Come talk to me if you're confused
        self.reset_visibility_cache()
        return self.step(
            action="__Teleport__",
            position=position,
            rotation=dict(x=0, y=rotation, z=0),
            **kwargs,
        )

    def reset_visibility_cache(self):
        self._visible_objects_cache = {
            "nav": {},
            "manip": {},
        }
        self.camera_registry.stepwise_reset()

    def get_camera_behind_object_view(self):
        """Get a camera view from behind the agent (for visualization/debugging)."""
        if len(self.controller.last_event.third_party_camera_frames) < 2:
            event = self.step(
                action="AddThirdPartyCamera",
                position=dict(x=0, y=2, z=-1),
                rotation=dict(x=50, y=0, z=0),
                fieldOfView=90,
                agentPositionRelativeCoordinates=True,
                parent="agent",
            )

        camera = self.controller.last_event.third_party_camera_frames[1]
        return camera

    def get_top_down_path_view(self, agent_path, targets_to_highlight=None):
        if len(self.controller.last_event.third_party_camera_frames) < 6:
            event = self.controller.step({"action": "GetMapViewCameraProperties"})
            cam = event.metadata["actionReturn"].copy()
            bounds = event.metadata["sceneBounds"]["size"]
            max_bound = max(bounds["x"], bounds["z"])

            cam["fieldOfView"] = 50
            cam["position"]["y"] += 1.1 * max_bound
            cam["orthographic"] = False
            cam["farClippingPlane"] = 50
            del cam["orthographicSize"]
            self.controller.step({"action": "AddThirdPartyCamera", "skyboxColor": "white", **cam})

        waypoints = []
        for target in targets_to_highlight or []:
            target_position = self.get_object_position(target)
            target_dict = {
                "position": target_position,
                "color": {"r": 1, "g": 0, "b": 0, "a": 1},
                "radius": 0.5,
                "text": "",
            }
            waypoints.append(target_dict)

        event = self.controller.step(
            {
                "action": "VisualizeWaypoints",
                "waypoints": waypoints,
            }
        )
        # put this over the waypoints just in case
        event = self.controller.step(
            {"action": "VisualizePath", "positions": agent_path, "pathWidth": 0.2}
        )
        self.controller.step({"action": "HideVisualizedPath"})
        path = event.third_party_camera_frames[-1]
        return path
        # cutoff = round(path.shape[1] * 6 / 396)  # yes this is ridiculous
        # return path[:, cutoff:-cutoff, :]

    def reset_agent_embodiment(
        self, initialize_cameras=False, randomize_embodiment=False, preset_camera_positions=False
    ):
        if not initialize_cameras:
            number_of_third_party_cameras = len(
                self.controller.last_event.metadata["thirdPartyCameras"]
            )
            assert (
                number_of_third_party_cameras == 1
            ), f"Expected 1 third party camera, got {number_of_third_party_cameras}"

        # Default Intel camera parameters
        base_y = 0.9009926 - 0.000999797135591507
        default_camera_positions = {
            "nav": {
                "position": {"x": 0.00192035, "y": 0.5447009, "z": 0.0678804},
                "rotation": {"x": 27.0, "y": 0.0, "z": 0.0},
                "fov": 59.0,
                "index": "main",
            },
            "manip": {
                "position": {"x": 0.05390513, "y": 0.5238336, "z": -0.05884857},
                "rotation": {"x": 33.0, "y": 90.0, "z": 0.0},
                "fov": 59.0,
                "index": 0,
            },
        }

        camera_positions = default_camera_positions.copy()
        if preset_camera_positions:
            for camera_name, preset_params in preset_camera_positions.items():
                camera_positions[camera_name] = {
                    "position": preset_params["commanded_pose"]["position"].copy(),
                    "rotation": preset_params["commanded_pose"]["rotation"].copy(),
                    "fov": preset_params["fov"],
                    "index": camera_positions[camera_name]["index"],
                }
                assert abs(camera_positions[camera_name]["fov"] - INTEL_VERTICAL_FOV) < 8, (
                    f"Input FOV {camera_positions[camera_name]['fov']} does not have enough correspondence to "
                    f"INTEL_VERTICAL_FOV - this may break critical functionality/assumptions"
                )

        for idx, (camera_name, camera_params) in enumerate(camera_positions.items()):
            third_party_action = (
                "AddThirdPartyCamera" if initialize_cameras else "UpdateThirdPartyCamera"
            )
            position = camera_params["position"].copy()
            rotation = camera_params["rotation"].copy()
            fov = camera_params["fov"]

            if randomize_embodiment:
                assert (
                    not preset_camera_positions
                ), "Cannot randomize embodiment with preset camera positions"
                position["x"] += np.random.uniform(-0.01, 0.01)
                position["y"] += np.random.uniform(-0.01, 0.01)
                position["z"] += np.random.uniform(-0.01, 0.01)
                rotation["x"] += np.random.uniform(-3, 3)
                rotation["y"] += np.random.uniform(-3, 3)
                rotation["z"] += np.random.uniform(-3, 3)
                fov += np.random.uniform(-5, 5)

            if camera_name == "nav":
                event = self.controller.step(
                    action="UpdateMainCamera",
                    position=position,
                    rotation=rotation,
                    fieldOfView=fov,
                    agentId=0,
                )
            else:
                step_kwargs = {
                    "action": third_party_action,
                    "position": position,
                    "rotation": rotation,
                    "fieldOfView": fov,
                    "parent": "agent",
                    "agentPositionRelativeCoordinates": True,
                }
                if not initialize_cameras:
                    step_kwargs["thirdPartyCameraId"] = camera_params["index"]
                else:
                    assert (
                        idx - 1 == camera_params["index"]
                    ), f"Cameras must be added in order, expected index {idx - 1}, got {camera_params['index']}"

                event = self.controller.step(**step_kwargs)
            self.camera_registry[camera_name].reset_commanded_position_and_rotation(
                position=position, rotation=rotation
            )

        # self.step(action="SetGripperOpenness", openness=30)
        self.camera_registry.task_reset(randomize_distortion_parameters=False)

    def reset(self, scene):
        if scene is None:
            raise ValueError("`scene` must be non-None.")

        self.current_scene_json = scene
        self.agent_ids = [i for (i, r) in AGENT_RADIUS_LIST]

        # add metadata here for navmesh?
        base_agent_navmesh = {
            "agentHeight": 1.8,
            "agentSlope": 10,
            "agentClimb": 0.5,
            "voxelSize": 0.1666667,
        }
        scene["metadata"]["navMeshes"] = [
            {**base_agent_navmesh, **{"id": i, "agentRadius": r}} for (i, r) in AGENT_RADIUS_LIST
        ]

        # Mostly for Phone2Proc scenes - may not work but will be corrected if possible in the scene reset.
        if "agent" not in scene["metadata"]:
            scene["metadata"]["agent"] = {
                "horizon": 30,
                "position": {"x": 0, "y": 0.95, "z": 0},
                "rotation": {"x": 0, "y": 270, "z": 0},
                "standing": True,
            }

        teleport_success = False
        used_alternative = False
        agent_source = scene["metadata"]["agent"]
        reset_event = None

        while not teleport_success:
            agent_source["horizon"] = HORIZON
            self.reset_visibility_cache()
            reset_event = self.controller.reset(scene=scene)
            self.reset_agent_embodiment(initialize_cameras=True, randomize_embodiment=False)
            if STRETCH_COMMIT_ID in [
                "3131f48169e8d48c1310fa83aac5c9c1a9cadde5",
                "df12ab257a9334f14a7d088019b570978cbbad4b",
            ]:
                self.controller.step(
                    "SetDefaultPhysicsSimulationParams",
                    defaultPhysicsSimulationParams=dict(
                        autoSimulation=self.initialization_args["autoSimulation"]
                    ),
                )
            # Do not display the unrealistic blue sphere on the agent's gripper
            # self.controller.step("ToggleMagnetVisibility", visible=False, raise_for_failure=True)

            self.set_object_filter([])
            self.room_poly_map, self.room_type_dict = get_rooms_polymap_and_type(
                self.current_scene_json
            )
            teleport_event = self.teleport_agent(**agent_source)

            if not teleport_event.metadata["lastActionSuccess"]:
                if (
                    "global_pose" in scene["metadata"]
                    and "agent" in scene["metadata"]["global_pose"]
                    and not used_alternative
                ):
                    print(
                        "FAILED TO TELEPORT AGENT AFTER INITIALIZATION. Retrying with alternative global_pose"
                    )
                    agent_source = scene["metadata"]["global_pose"]["agent"]
                    used_alternative = True
                else:
                    print("FAILED TO TELEPORT AGENT AFTER INITIALIZATION", scene)
                    return teleport_event
            else:
                teleport_success = True
        
        # evt = self.controller.step(
        #             action="DeleteLRUFromProceduralCache", 
        #             assetLimit=0
        #         )

        return reset_event

    def get_camera_parameters(
        self,
        which_camera: Literal["nav", "manip"],
    ):
        if self.camera_registry[which_camera]["index"] == "main":
            param_dict = dict(
                position=self.controller.last_event.metadata["agentPositionRelativeCameraPosition"],
                rotation=self.controller.last_event.metadata["agentPositionRelativeCameraRotation"],
                fov=self.controller.last_event.metadata["fov"],
                frame_height=self.controller.last_event.frame.shape[0],
                frame_width=self.controller.last_event.frame.shape[1],
                commanded_pose=self.camera_registry[which_camera]["commanded_pose"],
            )
        elif which_camera in self.camera_registry.cameras.keys():
            idx = self.camera_registry[which_camera]["index"]
            param_dict = dict(
                position=self.controller.last_event.metadata["thirdPartyCameras"][idx][
                    "agentPositionRelativeThirdPartyCameraPosition"
                ],
                rotation=self.controller.last_event.metadata["thirdPartyCameras"][idx][
                    "agentPositionRelativeThirdPartyCameraRotation"
                ],
                fov=self.controller.last_event.metadata["thirdPartyCameras"][idx]["fieldOfView"],
                frame_height=self.controller.last_event.third_party_camera_frames[idx].shape[0],
                frame_width=self.controller.last_event.third_party_camera_frames[idx].shape[1],
                commanded_pose=self.camera_registry[which_camera]["commanded_pose"],
            )

        else:
            raise NotImplementedError

        param_dict["camera_intrinsic"] = calc_camera_intrinsics(
            fov_y=param_dict["fov"],
            frame_height=param_dict["frame_height"],
            frame_width=param_dict["frame_width"],
        )

        return param_dict

    def get_visible_objects(
        self,
        which_camera: Literal["nav", "manip", "both"] = "nav",
        maximum_distance=2,
    ):
        # FYI: filtering by objects at this level has been removed to make best use
        # of the cache, but GetVisibleObjects still supports it with a list passed as objectIds=filter_object_ids.

        if which_camera == "both":
            return self.get_visible_objects("nav", maximum_distance) | self.get_visible_objects(
                "manip", maximum_distance
            )

        # on to the single-camera cases
        camera_info = self.camera_registry[which_camera]
        existing_cache = self._visible_objects_cache[which_camera]
        if maximum_distance in existing_cache:
            return existing_cache[maximum_distance]
        else:
            if self.camera_registry[which_camera]["index"] == "main":
                visible_objects = self.controller.step(
                    "GetVisibleObjects",
                    maxDistance=maximum_distance,
                ).metadata["actionReturn"]
            else:
                visible_objects = self.controller.step(
                    "GetVisibleObjects",
                    maxDistance=maximum_distance,
                    thirdPartyCameraIndex=camera_info["index"],
                ).metadata["actionReturn"]
            existing_cache[maximum_distance] = set(visible_objects)
            return existing_cache[maximum_distance]

    def get_approx_object_mask(
        self,
        object_id: str,
        which_camera: Literal["nav", "manip"],
        divisions: int,
    ):
        step_dict = dict(
            action="GetApproxObjectMask",
            objectId=object_id,
            divisions=divisions,
        )
        if self.camera_registry[which_camera]["index"] != "main":
            step_dict["thirdPartyCameraIndex"] = self.camera_registry[which_camera]["index"]
        return self.step(**step_dict).metadata["actionReturn"]

    def object_is_visible_in_camera(
        self,
        object_id,
        which_camera: Literal["nav", "manip", "both"] = "nav",
        maximum_distance=2,
    ):
        return object_id in self.get_visible_objects(
            which_camera=which_camera,
            maximum_distance=maximum_distance,
        )

    def get_objects(self) -> List[VIDAObject]:
        with self.include_object_metadata_context():
            return [VIDAObject(o) for o in self.controller.last_event.metadata["objects"]]

    def get_synset_and_pos_dict(self, uninteresting_synsets: Set[str]):
        all_obj = {}
        for obj in self.get_objects():
            if obj["synset"] in uninteresting_synsets:
                continue
            all_obj[obj["objectId"]] = {
                "type": obj["synset"],
                "pos": obj["position"],
            }
        return all_obj

    def set_object_filter(self, object_ids: List[str]):
        assert len(object_ids) == 0, "Please don't do this, talk to Luca about why."
        return self.controller.step(
            action="SetObjectFilter",
            objectIds=object_ids,
            raise_for_failure=True,
        )

    def reset_object_filter(self):
        return self.controller.step(action="ResetObjectFilter")

    @contextmanager
    def include_object_metadata_context(self):
        needs_reset = len(self.controller.last_event.metadata["objects"]) == 0
        try:
            if needs_reset:
                self.controller.step("ResetObjectFilter")
                assert self.controller.last_event.metadata["lastActionSuccess"]
            yield None
        finally:
            if needs_reset:
                obj_meta = self.controller.last_event.metadata["objects"]
                self.controller.step("SetObjectFilter", objectIds=[])
                self.controller.last_event.metadata["objects"] = obj_meta
                assert self.controller.last_event.metadata["lastActionSuccess"]

    def get_objects_that_objects_are_on(
        self, object_ids: Sequence[str]
    ) -> Dict[str, Optional[str]]:
        oid_to_on_oids = self.controller.step(
            action="CheckWhatObjectsOn",
            belowDistance=0.05,
            objectIds=object_ids,
            raise_for_failure=True,
        ).metadata["actionReturn"]

        on_oids = list(
            set(sum([on_oid for on_oid in oid_to_on_oids.values() if on_oid is not None], []))
        )

        on_oid_to_object = {None: None}
        if len(on_oids) != 0:
            on_oid_metadata = self.controller.step(
                action="GetMinimalObjectMetadata", objectIds=on_oids, raise_for_failure=True
            ).metadata["actionReturn"]
            on_oid_to_object.update({md["objectId"]: VIDAObject(md) for md in on_oid_metadata})

        return {
            oid: [on_oid_to_object[on_oid] for on_oid in on_oids]
            for oid, on_oids in oid_to_on_oids.items()
        }

    def get_object_receptacle_synsets(self, object_id: str):
        """
        THIS FUNCTION MAY BE SLOW IF CALLED AT EVERY STEP, perhaps use `get_objects_that_objects_are_on`
        instead?

        :param object_id:
        :return:
        """
        source_receptacle_ids = self.get_object(object_id, include_receptacle_info=True)[
            "parentReceptacles"
        ]

        if source_receptacle_ids is None:  # TODO why do we ever get into none?
            source_receptacle_ids = []

        source_receptacle_synsets = [
            self.get_object(obj_id, include_receptacle_info=True)["synset"]
            for obj_id in source_receptacle_ids
        ]
        return source_receptacle_synsets

    def get_locations_on_receptacle(self, receptacle_id, filter_by_quality=False):
        result = self.step(
            action="GetSpawnCoordinatesAboveReceptacle", objectId=receptacle_id, anywhere=True
        )
        result = result.metadata["actionReturn"]
        if not filter_by_quality:
            return result
        else:
            # Simple filtering: keep locations that are not too close to edges
            # This is a simplified version - for full implementation see stretch_controller.py
            filtered = []
            for loc in result:
                # Basic validation - can be enhanced if needed
                if all(k in loc for k in ["x", "y", "z"]):
                    filtered.append(loc)
            return filtered

    def get_current_agent_position(self):
        # TODO RMH USE STRETCH STATE INSTEAD at the usage point
        return StretchState(self.controller).base_position

    def get_current_agent_full_pose(self):
        # TODO RMH USE STRETCH STATE INSTEAD
        return {
            **self.controller.last_event.metadata["agent"],
            "arm": self.controller.last_event.metadata["arm"],
        }

    def query_env(self, **kwargs):
        # TODO RMH IS THIS ACTUALLY USED
        """
        :param kwargs: action, and other arguments to query the controller for information
        :return: Metadata from the environment
        """

        if "action" in kwargs:
            output = self.controller.step(**kwargs).metadata["actionReturn"]
        else:
            raise NotImplementedError
        return output

    def get_objects_of_synset_list(
        self,
        target_object_synsets: Iterable[str],
        include_hyponyms: bool,
        all_objs: Optional[List[VIDAObject]] = None,
    ):
        if all_objs is None:
            all_objs = self.get_objects()

        if include_hyponyms:
            return [
                vidaobj
                for vidaobj in all_objs
                if any(
                    is_hypernym_of(synset=vidaobj["synset"], possible_hypernym=other)
                    for other in target_object_synsets
                )
            ]
        else:
            return [vidaobj for vidaobj in all_objs if vidaobj["synset"] in target_object_synsets]

    def get_all_objects_of_synset(
        self, synset: str, include_hyponyms: bool, all_objs: Optional[List[VIDAObject]] = None
    ):
        return self.get_objects_of_synset_list(
            target_object_synsets=[synset], include_hyponyms=include_hyponyms, all_objs=all_objs
        )

    def get_available_object_synsets_from_synset_list(
        self,
        target_object_synsets: Iterable[str],
        include_hyponyms: bool,
        all_objs: Optional[List[VIDAObject]] = None,
    ) -> Set[str]:
        return {
            o["synset"]
            for o in self.get_objects_of_synset_list(
                target_object_synsets=target_object_synsets,
                include_hyponyms=include_hyponyms,
                all_objs=all_objs,
            )
        }

    def get_object(self, object_id: str, include_receptacle_info: bool = False):
        """
        NOTE: It may be much less efficient to `include_receptacle_info` than to not.

        :param object_id:
        :param include_receptacle_info:
        :return:
        """
        if include_receptacle_info or any(
            object_id == o["objectId"] for o in self.controller.last_event.metadata["objects"]
        ):
            with self.include_object_metadata_context():
                return VIDAObject(self.controller.last_event.get_object(object_id))

        meta = self.controller.step(
            action="GetObjectMetadata", objectIds=[object_id], raise_for_failure=True
        ).metadata["actionReturn"][0]

        del meta[
            "parentReceptacles"
        ]  # This will always be None when using GetObjectMetadata so remove it so there is no ambiguity
        return VIDAObject(meta)

    def get_obj_pos_from_obj_id(self, object_id):
        return self.get_object(object_id)["axisAlignedBoundingBox"]["center"]

    def get_object_position(self, object_id):
        try:
            return self.get_object(object_id)["position"]
        except:
            event = self.get_object(object_id)
            print(event)
            print(object_id)

    def get_agent_alignment_to_object(self, object_id: str, use_arm_orientation: bool = False):
        current_agent_pose = StretchState(self)
        alignment = agent_alignment_to_point(
            current_agent_pose, self.get_object_position(object_id), arm=use_arm_orientation
        )
        return alignment

    def get_agent_alignment_to_wall(
        self, wall_id, use_arm_orientation: bool = False, parsable_wall_id: Optional[str] = None
    ):
        current_agent_pose = StretchState(self)
        wall_location = get_wall_center_floor_level(
           parsable_wall_id or wall_id, y=current_agent_pose.base_position["y"]
        )
        return agent_alignment_to_point(current_agent_pose, wall_location, arm=use_arm_orientation)

    def get_reachable_positions(self, grid_size: Optional[float] = None):
        if grid_size is None:
            # Use a smaller grid size than the default as otherwise we may miss many
            # positions that are reachable when not moving with 90 degree rotations
            grid_size = GRID_SIZE * 0.75

        rp_event = self.controller.step(action="GetReachablePositions", gridSize=grid_size)
        if not rp_event:
            # NOTE: Skip scenes where GetReachablePositions fails
            warnings.warn(f"GetReachablePositions failed in {self.current_scene_json}")
            return []
        reachable_positions = rp_event.metadata["actionReturn"]
        return reachable_positions

    def get_touching_poses(
        self,
        object_id: str,
        positions: Optional[List[Vector3]] = None,
        max_distance: float = 1.0,
        max_poses: Optional[int] = None,
    ):
        other_action_kwargs = {}
        if max_poses is not None:
            other_action_kwargs["maxPoses"] = max_poses

        tp_event = self.controller.step(
            action="GetTouchingPoses",
            objectId=object_id,
            positions=positions or self.get_reachable_positions(),
            maxDistance=max_distance,
            **other_action_kwargs,
        )
        if not tp_event:
            warnings.warn(f"GetTouchingPoses failed")
            return []

        return tp_event.metadata["actionReturn"]

    def stop(self):
        self.controller.stop()

    def sufficient_agent_state_change(
        self, agent_state_before: StretchState, agent_state_after: StretchState
    ):
        # get the absolute value differences between the keys of the two states
        too_small, _ = StretchState.state_change_within_tolerance(
            delta_state=StretchState.difference(
                final_state=agent_state_after, initial_state=agent_state_before
            ),
            tolerance=self._universal_state_tolerance,
        )
        return not too_small

    def agent_step(self, action: StretchAction):
        agents_full_pose_before_action = StretchState(self.controller)

        action_dicts = action.enact()
        # this is a list of dicts that all resolve a dimension of motion in the environment
        # (e.g. base translate, arm up, wrist rotate, etc.)
        # For VIDA historically and currently, this is always a list of a single action
        # I have strong feelings about preserving the flexibility to actuate
        # multiple dimensions "at once" (given THOR limitations)
        # but I'm not absolutely fixated on this exact structure
        # let's talk
        all_sim_API_action_strings = [action_dict["action"] for action_dict in action_dicts]
        for action_dict in action_dicts:
            if action_dict["action"] in ["RotateWristRelative", "MoveArm", "MoveArmRelative"]:
                action_dict = {**action_dict, **ADDITIONAL_ARM_ARGS}
            elif action_dict["action"] == "MoveAgent":
                action_dict = {**action_dict, **ADDITIONAL_NAVIGATION_ARGS}

            event = self.step(**action_dict)

        if "ReleaseObject" in all_sim_API_action_strings:
            self.step(action="AdvancePhysicsStep", simSeconds=2)

        agents_full_pose_after_action = StretchState(self.controller)

        agent_moved = self.sufficient_agent_state_change(
            agents_full_pose_before_action, agents_full_pose_after_action
        )
        collision_in_error_message = "collided" in event.metadata["errorMessage"].lower()
        if isinstance(action, StretchGraspAction):
            if len(agents_full_pose_after_action.held_oids) > len(
                agents_full_pose_before_action.held_oids
            ):
                action_success = True
            else:
                action_success = False
        elif isinstance(action, StretchDropOffAction):
            action_success = True
        elif any(
            "arm" in action.lower() or "wrist" in action.lower()
            for action in all_sim_API_action_strings
        ):
            action_success = not collision_in_error_message and agent_moved
        else:
            action_success = not collision_in_error_message

        event.metadata["lastActionSuccess"] = action_success

        return event

    # calculate the shortest path to that location
    def get_shortest_path_to_object(
        self,
        object_id,
        initial_position=None,
        initial_rotation=None,
        specific_agent_meshes=None,
        attempt_path_improvement: bool = True,
    ) -> Optional[List[Vector3]]:
        """
        Computes the shortest path to an object from an initial position using a controller

        :param object_id: string with id of the object
        :param initial_position: dict(x=float, y=float, z=float) with the desired initial rotation
        :param initial_rotation: dict(x=float, y=float, z=float) representing rotation around axes or None
        :return:
        """
        mesh_restriction = specific_agent_meshes is not None
        if specific_agent_meshes is None:
            specific_agent_meshes = self.agent_ids

        if initial_position is None:
            initial_position = self.get_current_agent_position()

        for nav_mesh_id in specific_agent_meshes:
            args = dict(
                action="GetShortestPath",
                objectId=object_id,
                position=initial_position,
                navMeshIds=[nav_mesh_id],  # update to incorporate navmesh
            )
            if initial_rotation is not None:
                args["rotation"] = initial_rotation
            event = self.step(**args)
            if event.metadata["lastActionSuccess"]:
                corners = event.metadata["actionReturn"]["corners"]
                if len(corners) == 0:
                    continue

                if (
                    nav_mesh_id > 1
                    and not mesh_restriction
                    and attempt_path_improvement
                    and len(corners) > 4
                ):
                    corners = self.split_and_replan_paths(
                        initial_position, corners[-1], corners, recursion_depth=1
                    )
                self.last_successful_path = corners

                if attempt_path_improvement and len(corners) > 2:
                    corners = snap_to_skeleton(
                        controller=self,
                        corners=corners,
                    )

                return corners  # This will slow down data generation

        return None

    def does_some_shortest_path_to_object_exist(
        self,
        object_id: str,
        initial_position=None,
        initial_rotation=None,
    ) -> bool:
        """
        Checks if a shortest path to an object from an initial position exists. This is faster than
        `get_shortest_path_to_object` as we will only use the most general nav mesh.

        :param object_id: string with id of the object
        :param initial_position: dict(x=float, y=float, z=float) with the desired initial rotation
        :param initial_rotation: dict(x=float, y=float, z=float) representing rotation around axes or None
        :return:
        """
        return (
            self.get_shortest_path_to_object(
                object_id=object_id,
                initial_position=initial_position,
                initial_rotation=initial_rotation,
                specific_agent_meshes=[self.agent_ids[-1]],
                attempt_path_improvement=False,
            )
            is not None
        )

    def split_and_replan_paths(self, initial_position, target_position, path, recursion_depth=0):
        first_half = path[: (len(path) // 2)]
        second_half = path[(len(path) // 2) :]

        # Recursive call to get_shortest_path_to_point for each half
        first_half_replan = self.get_shortest_path_to_point(
            first_half[-1], initial_position, recursion_depth=recursion_depth
        )
        if first_half_replan is not None:
            first_half = first_half_replan

        second_half_replan = self.get_shortest_path_to_point(
            target_position, first_half_replan[-1], recursion_depth=recursion_depth
        )
        if second_half_replan is not None:
            second_half = second_half_replan
        return first_half + second_half[1:]

    # calculate the shortest path to that location
    def get_shortest_path_to_point(
        self,
        target_position,
        initial_position=None,
        initial_rotation=None,
        specific_agent_meshes=None,
        attempt_path_improvement=True,
        recursion_depth=0,
    ):
        """
        Computes the shortest path to an object from an initial position using a controller
        :param controller: agent controller
        :param object_id: string with id of the object
        :param initial_position: dict(x=float, y=float, z=float) with the desired initial rotation
        :param initial_rotation: dict(x=float, y=float, z=float) representing rotation around axes or None
        :return:
        """
        mesh_restriction = specific_agent_meshes is not None
        if specific_agent_meshes is None:
            specific_agent_meshes = self.agent_ids
        if initial_position is None:
            initial_position = self.get_current_agent_position()

        for nav_mesh_id in specific_agent_meshes:
            args = dict(
                action="GetShortestPathToPoint",
                position=initial_position,
                target=target_position,
                navMeshIds=[nav_mesh_id],  # update to incorporate navmesh
            )
            if initial_rotation is not None:
                args["rotation"] = initial_rotation
            event = self.step(**args)
            if event.metadata["lastActionSuccess"]:
                corners = event.metadata["actionReturn"]["corners"]
                if len(corners) == 0:
                    continue
                if (
                    nav_mesh_id > 1
                    and not mesh_restriction
                    and attempt_path_improvement
                    and len(corners) > 4
                    and recursion_depth < 3
                ):
                    corners = self.split_and_replan_paths(
                        initial_position, target_position, corners, recursion_depth + 1
                    )

                self.last_successful_path = corners

                if attempt_path_improvement and len(corners) > 2:
                    corners = snap_to_skeleton(
                        controller=self,
                        corners=corners,
                    )

                return corners  # This will slow down data generation

        return None

    # def num_pixels_visible(self, object_id, manipulation_camera=False):
    #     assert (
    #         "renderInstanceSegmentation" in self.initialization_args
    #         and self.initialization_args["renderInstanceSegmentation"]
    #     )
    #     if manipulation_camera:
    #         masks = self.manipulation_camera_segmentation
    #     else:
    #         masks = self.navigation_camera_segmentation
    #
    #     if object_id not in masks:
    #         return 0
    #
    #     mask = masks[object_id]
    #     return mask.sum()

    def is_object_visible_enough_for_interaction(self, object_id: str):
        return is_any_object_sufficiently_visible_and_in_center_frame(
            controller=self,
            object_ids=[object_id],
            align_for_manipulation=True,
            object_synset=None,
        )
        
    @property
    def navigation_camera_segmentation(
        self,
    ):
        if self.controller.last_event.instance_segmentation_frame is None:
            self.controller.step("Pass", renderImageSynthesis=True)
            assert self.controller.last_event.instance_segmentation_frame is not None, (
                "Must pass `renderInstanceSegmentation=True` on initialization"
                " to obtain a navigation_camera_segmentation"
            )

        return self.controller.last_event.instance_masks

    def get_closest_object_from_ids(self, object_ids, return_id_and_dist: bool = False):
        all_paths = [
            (
                obj_id,
                self.get_shortest_path_to_object(
                    obj_id,
                    specific_agent_meshes=[self.agent_ids[-1]],
                    attempt_path_improvement=False,
                ),
            )
            for obj_id in object_ids
        ]

        min_dist = float("inf")
        closest_obj_id = None
        for obj_id, path in all_paths:
            if path is None:
                continue
            dist = sum_dist_path(path)
            if dist < min_dist:
                min_dist = dist
                closest_obj_id = obj_id
        return closest_obj_id if not return_id_and_dist else (closest_obj_id, min_dist)

    def get_candidate_points_in_room(
        self,
        room_id,
        room_triangles: Optional[GeometryCollection] = None,
    ):
        polygon = self.room_poly_map[room_id]

        if room_triangles is None:
            # Triangulates the room, and takes the centers of all triangles as possible
            # target locations
            room_triangles = triangulate_room_polygon(polygon)

        candidate_points = [
            ((t.centroid.x, t.centroid.y), t.area) for t in room_triangles  # type:ignore
        ]

        # We sort the triangles by size so we try to go to the center of the largest triangle first
        candidate_points.sort(key=lambda x: x[1], reverse=True)
        candidate_points = [p[0] for p in candidate_points]

        # The centroid of the whole room polygon need not be in the room when the room is concave. If it is,
        # let's make it the first point we try to navigate to.
        if polygon.contains(polygon.centroid):
            candidate_points.insert(0, (polygon.centroid.x, polygon.centroid.y))

        candidate_points = [
            p for p in candidate_points if self.room_poly_map[room_id].contains(Point(p))
        ]

        return candidate_points

    def get_shortest_path_to_room_candidate_points(
        self,
        candidate_points,
        specific_agent_meshes=None,
        max_tries: int = 5,
        use_largest_possible_mesh: bool = False,
    ):
        assert max_tries > 0

        current_agent_position = self.controller.last_event.metadata["agent"]["position"]
        y = current_agent_position["y"]

        if specific_agent_meshes is None:
            specific_agent_meshes = self.agent_ids
        specific_agent_meshes = sorted(specific_agent_meshes)

        path = None
        for agent_id in specific_agent_meshes:
            for point in candidate_points[:max_tries]:
                path = self.get_shortest_path_to_point(
                    target_position=dict(x=point[0], y=y, z=point[1]),
                    initial_position=current_agent_position,
                    specific_agent_meshes=[agent_id],
                    attempt_path_improvement=False,
                )
                if path is not None:
                    break
            if use_largest_possible_mesh and path is not None:
                break
        return path

    def get_shortest_path_to_room(
        self,
        room_id,
        specific_agent_meshes=None,
        max_tries: int = 5,
        room_triangles: Optional[GeometryCollection] = None,
    ):
        candidate_points = self.get_candidate_points_in_room(
            room_id=room_id,
            room_triangles=room_triangles,
        )

        return self.get_shortest_path_to_room_candidate_points(
            candidate_points, specific_agent_meshes, max_tries
        )

    def get_objects_room_id_and_type(self, object_id: str) -> Tuple[str, str]:
        object_position = self.get_object_position(object_id)
        room_id = get_room_id_from_location(self.room_poly_map, object_position)
        room_type_return = (
            self.room_type_dict[room_id] if room_id is not None else None
        )  # making it more robust to none style cases
        return room_id, room_type_return

    def find_closest_room_of_list(self, room_ids, return_id_and_dist: bool = False):
        all_paths = []
        for room_id in room_ids:
            path = self.get_shortest_path_to_room(
                room_id, specific_agent_meshes=[self.agent_ids[-1]]
            )
            all_paths.append((room_id, path))

        min_dist = float("inf")
        closest_room_id = None
        for room_id, path in all_paths:
            if path is None:
                continue
            dist = sum_dist_path(path)
            if dist < min_dist:
                min_dist = dist
                closest_room_id = room_id

        return closest_room_id if not return_id_and_dist else (closest_room_id, min_dist)

    def get_current_scene_json(self):
        return self.current_scene_json

    def get_agent_dist_from_room_ids(
        self,
        rooms,
    ):
        room_ids = [room["id"] for room in rooms]

        room_ids_json = [room["id"] for room in self.current_scene_json["rooms"]]

        all_paths = []
        for room_id in room_ids:
            path = self.get_shortest_path_to_room(
                room_id, specific_agent_meshes=[self.agent_ids[-1]]
            )
            all_paths.append((room_id, path))

        to_return = {}
        for room_id, path in all_paths:
            if path is None:
                continue
            to_return[room_id] = sum_dist_path(path)

        return to_return

