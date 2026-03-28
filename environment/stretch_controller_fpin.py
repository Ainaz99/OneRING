import copy
import math
import os
import pdb
import random
import warnings
from collections import deque
from contextlib import contextmanager
from typing import Dict, Optional, Set, Sequence, List, Tuple, Iterable, Literal, Union
from typing import Union
from ai2thor.server import Event
import numpy as np
from skimage.morphology import remove_small_objects, binary_opening
from scipy.ndimage import label
from skimage.transform import resize
import torch
from ai2thor.controller import Controller
from shapely import Polygon, GeometryCollection, Point
import time


from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from environment.action_spaces import agent_alignment_to_point
from environment.actions import StretchAction, StretchGraspAction, StretchDropOffAction
from environment.stretch_state import StretchState
from environment.agent_parameter_utils import (
    AGENT_BODY_PARAMS,
    AgentParamRandomizer,
    AgentParams,
    get_nav_mesh_config_from_box,
)
from utils.constants.FPIN_utils import CUTOFF
from environment.vida_objects import VIDAObject
from utils.constants.FPIN_utils import ALLOW_CONSECUTIVE_FAILURES

from utils.constants.stretch_initialization_utils import (
    AGENT_COLLIDER_NAVMESH_REL_SCALE,
    AGENTS_BASE_HEIGHT,
    ADDITIONAL_ARM_ARGS,
    HORIZON,
    ADDITIONAL_NAVIGATION_ARGS,
    STRETCH_COMMIT_ID,
    GRID_SIZE,
    assert_agent_objaverse_mesh_exists,
)

from utils.constants.FPIN_utils import DEFAULT_ASSET, PARAMS_TO_RANDOMIZE

from utils.data_generation_utils.exception_utils import (
    AgentIsStuckException,
)
from utils.distance_calculation_utils import sum_dist_path, position_dist
from environment.action_spaces import agent_alignment_to_point
from utils.data_generation_utils.navigation_utils import (
    get_rooms_polymap_and_type,
    get_room_id_from_location,
    get_wall_center_floor_level,
    triangulate_room_polygon,
    is_any_object_sufficiently_visible_and_in_center_frame,
    snap_to_skeleton,
)

from utils.distance_calculation_utils import sum_dist_path, position_dist
from utils.synsets.hypernyms import is_hypernym_of
from utils.type_utils import Vector3



class FPINController:

    def __init__(self, initialize_controller=True, store_action_trace=False, **kwargs):
        self.list_of_last_action_success = deque(maxlen=20)
        self.should_render_image_synthesis = (
            kwargs.get("renderDepthImage", False)
            or kwargs.get("renderNormalsImage", False)
            or kwargs.get("renderFlowImage", False)
        )
        self.mode = None

        self.store_action_trace = store_action_trace
        self.trace_sequence = 0
        self.trace = {}  # OrderedDict()


        self.agent_segm_masks = {"nav": None, "manip": None}

        
        if DEFAULT_ASSET == "ghost":
            print(f"params_to_randomize: {PARAMS_TO_RANDOMIZE}")
            self.agent_param_randomizer = kwargs.get(
                "agent_param_randomizer",
                AgentParamRandomizer(parameters_to_randomize=PARAMS_TO_RANDOMIZE, agent_asset=DEFAULT_ASSET)
                # AgentParamRandomizer(parameters_to_randomize="ALL") #TODO KIANA
            )

        elif DEFAULT_ASSET == "locobot":
            self.agent_param_randomizer = AgentParamRandomizer(
                    exact_values=dict(
                        first_camera_fov= 42, #60
                        width=396, #400,
                        height=224, #300,
                        first_camera_rotation={"x": 0.0, "y": 0.0, "z": 0.0},
                        first_camera_position={"x": 0.0, "y": +0.9009997844696045-0.000999797135591507-0.0312, "z": 0.0},
                    ),
                    agent_asset='locobot',
                )


        elif DEFAULT_ASSET == "unitree_a1":
            self.agent_param_randomizer = AgentParamRandomizer(
                    exact_values=dict(
                        first_camera_fov=42,
                        width=480,
                        height=270,
                        first_camera_rotation={"x": 0.0, "y": 0.0, "z": 0.0}, 
                        first_camera_position={"x": 0.0091, "y": 0.2998, "z": 0.2723}
                    ),
                agent_asset='unitree_a1',
            )


        elif DEFAULT_ASSET == "stretch":
            self.agent_param_randomizer = AgentParamRandomizer(
                    exact_values=dict(
                        first_camera_fov=59,
                        second_camera_fov=59,
                        width=396,
                        height=224,
                        first_camera_rotation={"x": 27, "y": 0.0, "z": 0.0}, #27,
                        second_camera_rotation={"x": 33, "y": 90.0, "z": 0.0}, #33,
                        first_camera_position={"x": 0.00192035, "y": +0.9009926-0.000999797135591507+0.5447009, "z": 0.0678804},
                        second_camera_position={"x": 0.05390513, "y": 0.9009926-0.000999797135591507+0.5238336, "z": -0.05884857},
                    ),
                    agent_asset='stretch',
                )


        if "agent_param_randomizer" in kwargs:
            del kwargs["agent_param_randomizer"]
        self.agent_asset = self.agent_param_randomizer.agent_asset

        self.room_poly_map: Optional[Dict[str, Polygon]] = None
        self.room_type_dict: Optional[Dict[str, str]] = None

        self.agent_params = None
        self.controller = None

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
        self._nav_visible_objects_cache = {}
        self._manip_visible_objects_cache = {}

        if initialize_controller:
            self.agent_segm_masks = {"nav": None, "manip": None}

            full_list = {
                **kwargs,
                "agentInitializationParams": AGENT_BODY_PARAMS[self.agent_asset],
            }
            specific_scene = None
            if "scene" not in kwargs:
                full_list["scene"] = "Procedural"
            else:
                specific_scene = kwargs["scene"]
                del kwargs["scene"]

            assert_agent_objaverse_mesh_exists(full_list["agentInitializationParams"])

            trace_id = self.add_step_trace(full_list, call="constructor")
            self.controller = Controller(**full_list)
            self.add_step_result_trace(self.controller.last_event, trace_id)

            self.initialization_args = kwargs
            print(f"Using Controller commit id: {self.controller._build.commit_id}")
            assert STRETCH_COMMIT_ID in self.controller._build.commit_id  # TODO KIANA

            if "scene" in kwargs:
                self.reset(kwargs["scene"])

        


    def get_controller_camera_params(self, which_camera: Literal["nav", "manip"]):
        if which_camera == "nav":
            camera_rel_position = self.controller.last_event.metadata[
                "agentPositionRelativeCameraPosition"
            ]
            camera_rel_rotation = self.controller.last_event.metadata[
                "agentPositionRelativeCameraRotation"
            ]
            fov_y = self.controller.last_event.metadata["fov"]
        elif which_camera == "manip":
            camera_rel_position = self.controller.last_event.metadata["thirdPartyCameras"][0][
                "agentPositionRelativeThirdPartyCameraPosition"
            ]
            camera_rel_rotation = self.controller.last_event.metadata["thirdPartyCameras"][0][
                "agentPositionRelativeThirdPartyCameraRotation"
            ]
            fov_y = self.controller.last_event.metadata["thirdPartyCameras"][0]["fieldOfView"]
        else:
            raise ValueError(f"Invalid camera: {which_camera}")

        return camera_rel_position, camera_rel_rotation, fov_y


    def get_object_from_pixel(self, point: List[int], which_camera: Literal["nav", "manip"]):
        if which_camera == "nav":
            frame = self.navigation_camera
            masks_to_look_at = self.navigation_camera_segmentation
        elif which_camera == "manip":
            frame = self.manipulation_camera
            masks_to_look_at = self.manipulation_camera_segmentation
        else:
            raise ValueError(f"Invalid camera: {which_camera}")

        assert (
            point[0] < frame.shape[0]
            and point[1] < frame.shape[1]
            and point[0] >= 0
            and point[1] >= 0
        ), f"Point {point} is not in frame shape {frame.shape}"
        for obj_id in masks_to_look_at:
            if masks_to_look_at[obj_id][point[0], point[1]]:
                return obj_id
        raise ValueError(f"No object found at pixel {point}")

    def get_agent_parameters(self) -> AgentParams:
        return self.agent_params
    
    def get_objects_in_hand_sphere(self):
        # return self.controller.last_event.metadata["arm"]["pickupableObjects"]
        return []

    def get_held_objects(self):
        return []
        # return self.controller.last_event.metadata["arm"]["heldObjects"]

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
        frame = self.controller.last_event.frame
        # cutoff = round(frame.shape[1] * 6 / 396) # TODO not for this project
        return frame[:, CUTOFF:-CUTOFF, :]

    def get_cutoff_amount(self):
        frame = self.controller.last_event.frame
        cutoff = round(frame.shape[1] * 6 / 396)
        return cutoff



    @property
    def manipulation_camera(self):
        frame = self.controller.last_event.third_party_camera_frames[0]
        # cutoff = round(frame.shape[1] * 6 / 396) # TODO not for this project
        return frame[:, CUTOFF:-CUTOFF, :3]  # TODO why do we have to do this :3

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

    @property
    def manipulation_camera_segmentation(
        self,
    ):  # TODO KE: THIS IS NOT CROPPED, USE get_segmentation_mask_of_object INSTEAD OR BE CAREFUL
        if self.controller.last_event.instance_segmentation_frame is None:
            self.controller.step("Pass", renderImageSynthesis=True)
            assert self.controller.last_event.instance_segmentation_frame is not None, (
                "Must pass `renderInstanceSegmentation=True` on initialization"
                " to obtain a manipulation_camera_segmentation"
            )

        return self.controller.last_event.third_party_instance_masks[0]
    
    

    @property
    def manipulation_depth_frame(self):
        frame = self.controller.last_event.third_party_depth_frames[0]
        # cutoff = round(frame.shape[1] * 6 / 396) # TODO not for this project
        return frame[:, CUTOFF:-CUTOFF]

    @property
    def navigation_depth_frame(self):
        frame = self.controller.last_event.depth_frame
        # cutoff = round(frame.shape[1] * 6 / 396) # TODO not for this project
        return frame[:, CUTOFF:-CUTOFF]

    @property
    def navigation_full_segmentation_frame(self):
        segm = self.navigation_camera_segmentation
        frame = self.generate_full_mask_frame(segm)
        return frame[:, CUTOFF:-CUTOFF]

    @property
    def manipulation_full_segmentation_frame(self):
        segm = self.manipulation_camera_segmentation
        frame = self.generate_full_mask_frame(segm)
        return frame[:, CUTOFF:-CUTOFF]

    def generate_full_mask_frame(self, segm):
        full_frame = None
        for key in segm:
            if full_frame is None:
                full_frame = np.zeros((segm[key].shape[0], segm[key].shape[1], 3), dtype=np.uint8)
            random_color = np.random.randint(0, 255, 3)
            full_frame[segm[key]] = random_color

        if full_frame is None:
            assert len(segm.keys()) == 0
            return np.zeros(self.controller.last_event.frame.shape).astype(np.uint8)

        return full_frame
    
    def process_agent_mask(self, mask):
        # mask has some artifacts, for example gaps between objects
        # Remove small objects
        cleaned_mask = remove_small_objects(mask, min_size=500)
        # Binary opening to remove thin lines
        cleaned_mask = binary_opening(cleaned_mask)
        # Label connected components
        labeled_mask, num_labels = label(cleaned_mask)
        # Keep the largest connected component (assuming it’s the agent)
        if num_labels > 1:
            largest_component = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
            cleaned_mask = (labeled_mask == largest_component)

        if mask.ndim == 2:
            mask = np.repeat(cleaned_mask[:, :, np.newaxis], 3, axis=-1)
        return mask.astype(float)

    
    def get_agent_segm_mask(self, camera):
        # if we have the mask 
        if self.agent_segm_masks[camera] is not None:
            return self.agent_segm_masks[camera]
        if camera == "nav":
            mask = np.all((self.navigation_full_segmentation_frame == [0,0,0]), axis=-1)
        elif camera == "manip":
            mask = np.all((self.manipulation_full_segmentation_frame == [0,0,0]), axis=-1)
        else:
            raise ValueError(f"Invalid camera type: {self.camera})")
        
        self.agent_segm_masks[camera] = self.process_agent_mask(mask)
        return self.agent_segm_masks[camera]

    def get_segmentation_mask_of_object(
        self, object_id: str, which_camera: Literal["nav", "manip"]
    ):
        if which_camera == "nav":
            segmentation_to_look_at = self.navigation_camera_segmentation
        elif which_camera == "manip":
            segmentation_to_look_at = self.manipulation_camera_segmentation
        else:
            raise NotImplementedError

        if object_id in segmentation_to_look_at:
            mask = segmentation_to_look_at[object_id]
            cutoff = round(mask.shape[1] * 6 / 396)
            result = mask[:, CUTOFF:-CUTOFF]
            assert result.shape == self.navigation_camera.shape[:2]
            return result
        else:
            return np.zeros(self.navigation_camera.shape[:2], dtype=bool)

    def get_relative_stretch_current_arm_state(self):
        return dict(x=0, y=0, z=0)

    def reset_step_trace(self):
        self.trace_sequence = 0
        self.trace = dict()

    def add_step_result_trace(self, event, trace_id):
        if self.store_action_trace:

            trace_id = int(trace_id)
            if trace_id in self.trace:
                self.trace[int(trace_id)]["event"] = dict(
                    traceId=trace_id,
                    lastActionSuccess=event.metadata["lastActionSuccess"],
                    errorMessage=event.metadata["errorMessage"],
                    actionReturn=event.metadata["actionReturn"],
                    lastAction=self.controller.last_action["action"],
                    # afterEventSequenceId=self.controller.server.sequence_id
                )
            else:
                print(
                    f"Error: could not store trace result for {trace_id}. store_action_trace was not called before for {trace_id}"
                )

    def add_step_trace(self, action_args, call="step"):

        if self.store_action_trace:
            trace_id = self.trace_sequence

            # action_key = "step" if not is_reset else "reset"
            # action_key = call
            # print(f"======= store_step_trace: trace: {trace_id} seq: {sequence_id} {action_key} {action_args['action'] if action_key=='step' else '' }")

            # trace = dict(step=action_args)
            # print(self.trace)
            self.trace[trace_id] = dict(call=call)
            # self.trace[trace_id]["call"] = call
            self.trace[trace_id]["args"] = action_args

            self.trace_sequence += 1
            return trace_id
        return None

    def dump_trace(self, out_filepath, indent=4, thor_ready=False):
        if self.store_action_trace:
            import json

            with open(out_filepath, "w") as f:
                if not thor_ready:
                    sequence_ids = self.trace.keys()  # sorted(self.trace.keys())

                    # json.dump([self.trace[key] for key in sequence_ids], f, indent=indent)
                    for key in self.trace.keys():
                        args = self.trace[key]["args"]

                        if "action_hook_runner" in args:
                            # print(vars(args["action_hook_runner"]))
                            args["action_hook_runner"] = args["action_hook_runner"].__dict__

                            constructor_keys = set(
                                [
                                    "asset_directory",
                                    "target_dir",
                                    "asset_symlink",
                                    "load_file_in_unity",
                                    "stop_if_fail",
                                    "asset_limit",
                                    "extension",
                                    "verbose",
                                ]
                            )

                            to_delete = [
                                key
                                for key in args["action_hook_runner"].keys()
                                if key not in constructor_keys
                            ]
                            for key in to_delete:
                                del args["action_hook_runner"][key]
                            # pass
                        if "server_class" in args:
                            args["server_type"] = args["server_class"].server_type
                            del args["server_class"]

                    json_out = [dict(trace_id=key, **self.trace[key]) for key in sequence_ids]
                    # print(json_out)
                    # print(json_out[0])

                    json.dump(json_out, f, indent=indent)
                else:
                    sequence_ids = sorted(self.trace.keys())
                    json.dump([self.trace[key]["step"] for key in sequence_ids], f, indent=indent)
                    # with open(os.path.join(out_filepath,))

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
            kwargs["action"] = "Teleport"

            

        seq_id = self.add_step_trace(kwargs)
        event = self.controller.step(**kwargs)
        self.add_step_result_trace(event, trace_id=seq_id)

        # # TODO KE: just a hack for thor bug
        if "action" in kwargs and "Teleport" in kwargs["action"]:
            # TODO KE: SUPER HACKY JUST FOR THOR BUG
            if event.metadata["lastActionSuccess"] == False:
                fixable_error = "> 0.05) in the y component"
                if fixable_error in event.metadata["errorMessage"] or True:
                    new_args = copy.deepcopy(kwargs)
                    new_args['position']["y"] = 0.05

                    seq_id = self.add_step_trace(kwargs)
                    event = self.controller.step(**new_args)
                    self.add_step_result_trace(event, trace_id=seq_id)
                    
        
            # TODO KE: THIS IS SO FUCKING HACKY. JUST FOR THE BUG
            self.calibrate_agent(self.agent_params)
        

        return event

    def get_camera_behind_object_view(self):
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
        return camera[:, CUTOFF:-CUTOFF, :]
    
    def teleport_agent(self, position: Vector3, rotation: Union[Vector3, float], **kwargs) -> Event:
        if isinstance(rotation, Dict):
            rotation = rotation["y"]

        if "standing" in kwargs:
            del kwargs["standing"]

        if "horizon" in kwargs:
            del kwargs["horizon"]
            # warnings.warn(
            #     "`horizon` is not a valid argument for teleport_agent, as camera locations are set on reset."
            #     " This argument will be ignored."
            # )

        if len(kwargs) > 0:
            allowed_keys = {
                "forceAction",
                "renderImage",
                "renderImageSynthesis",
                "raise_for_failure",
                "agentId",
            }
            assert set(kwargs.keys()).issubset(
                allowed_keys
            ), f"Invalid arguments for teleport_agent: {set(kwargs.keys()) - allowed_keys}"

        return self.step(
            action="__Teleport__",
            position=position,
            rotation=dict(x=0, y=rotation, z=0),
            **kwargs,
        )


    def reset_visibility_cache(self):
        self._nav_visible_objects_cache = {}
        self._manip_visible_objects_cache = {}

    def get_top_down_path_view(self, agent_path, targets_to_highlight=None):
        if len(self.controller.last_event.third_party_camera_frames) < 2:
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
        # cutoff = round(path.shape[1] * 6 / 396)  # yes this is ridiculous
        return path[:, CUTOFF:-CUTOFF, :]

    def calibrate_agent(self, agent_params: AgentParams):

        if len(self.controller.last_event.metadata["thirdPartyCameras"]) == 0:
            self.step(
                action="AddThirdPartyCamera",
                position=dict(x=-1.25, y=1, z=-1),
                rotation=dict(x=90, y=0, z=0),
                fieldOfView=90,
                agentPositionRelativeCoordinates=True,
                parent="agent",
            )

        # KE: THIS can be cleaner once we make ai2 api better
        assert (
            len(self.controller.last_event.metadata["thirdPartyCameras"]) <= 2
            and len(self.controller.last_event.metadata["thirdPartyCameras"]) > 0
        )


        manipulation_camera = self.controller.last_event.metadata["thirdPartyCameras"][0]
        thirdPartyCameraId = manipulation_camera["thirdPartyCameraId"]
        second_camera_position = agent_params.second_camera_position
        second_camera_rotation = agent_params.second_camera_rotation
        second_camera_fov = agent_params.second_camera_fov

        first_camera_position = agent_params.first_camera_position
        first_camera_rotation = agent_params.first_camera_rotation
        first_camera_fov = agent_params.first_camera_fov


        update_manip_cam = self.step(
            action="UpdateThirdPartyCamera",
            position=second_camera_position,
            rotation=second_camera_rotation,
            thirdPartyCameraId=thirdPartyCameraId,
            fieldOfView=second_camera_fov,
            agentPositionRelativeCoordinates=True,
            parent="agent",
        )
        update_nav_cam = self.step(
            action="UpdateMainCamera",
            position=first_camera_position,
            rotation=first_camera_rotation,
            fieldOfView=first_camera_fov,
        )

        all_events = [
            update_manip_cam,
            update_nav_cam,
        ]
        if not all([e.metadata["lastActionSuccess"] for e in all_events]):
            print("FAILED TO CALIBRATE AGENT")
            print(
                "[update_manip_cam, update_nav_cam, ",
                [
                    update_manip_cam,
                    update_nav_cam,
                ],
            )

    def get_navmeshes(
        self,
        box_body_sizes,
        base_collider_scale=dict(x=1, y=1, z=1),
        min_side_as_radius=True,
        radius_epsilon=0.0,
    ):
        return [
            get_nav_mesh_config_from_box(
                base_collider_scale=base_collider_scale,  # collider scale is already baked in fpinColliderSize return
                nav_mesh_scale=nav_mesh_scale,
                box_body_sizes=box_body_sizes,
                nav_mesh_id=nav_mesh_id,
                min_side_as_radius=min_side_as_radius,
                radius_epsilon=radius_epsilon,
            )
            for (nav_mesh_id, nav_mesh_scale) in AGENT_COLLIDER_NAVMESH_REL_SCALE
        ]

    def reset(
        self,
        scene,
        create_navmesh_from_initialize_sizes=False,
        navmesh_radius_eps=0.0,
        reset_agent_params=None,
    ):

        if scene is None:
            raise ValueError("`scene` must be non-None.")
        if reset_agent_params:
            self.agent_params = AgentParams.from_dict(reset_agent_params)
            
        self.current_scene_json = scene
        self.agent_ids = [i for (i, r) in AGENT_COLLIDER_NAVMESH_REL_SCALE]
        if self.agent_params is None:
            self.agent_params = self.agent_param_randomizer.get_random_agent_param()

        box_body_sizes = self.agent_params.get_body_size_after_load(agent_asset=self.agent_asset)
        

        # TODO remove this, I would not use the cache of sizes for risking outdated code
        if not create_navmesh_from_initialize_sizes:
            scene["metadata"]["navMeshes"] = self.get_navmeshes(
                base_collider_scale=self.agent_params.get_collider_scale(),
                box_body_sizes=box_body_sizes,
                min_side_as_radius=True,
                radius_epsilon=navmesh_radius_eps,
            )
            self.navmeshes = scene["metadata"]["navMeshes"]
            


        # Mostly for Phone2Proc scenes - may not work but will be corrected if possible in the scene reset.
        if "agent" not in scene["metadata"]:
            scene["metadata"]["agent"] = {
                "horizon": HORIZON,
                "position": {"x": 0, "y": 0.95, "z": 0},
                "rotation": {"x": 0, "y": 270, "z": 0},
                "standing": True,
            }

        scene["metadata"]["agent"]["horizon"] = HORIZON
        
        self.reset_visibility_cache() # TODO Ainaz: do we need this?
        

        ###############################################
        # reset the agent with random parameters
        ###############################################
        reset_params = dict(scene="Procedural")

        height = self.agent_params.height
        width = self.agent_params.width

        print(f"height: {height}, width: {width}")
        print(f"agent_params: {self.agent_params}")
        
        reset_params = {
            **reset_params,
            **{"height": height, "width": width},
            # **{"agentInitializationParams": AGENT_BODY_PARAMS[self.agent_asset]},
            **{"agentInitializationParams": self.agent_params.agent_body_params},
        }
        

        try:
            seq_id = self.add_step_trace(reset_params, call="reset")
            reset_event = self.controller.reset(**reset_params)
            self.add_step_result_trace(reset_event, trace_id=seq_id)

        except Exception as e:
            error_message = str(e)
            raise e
        

        if create_navmesh_from_initialize_sizes:
            # This is metadata after initializing the agent and what initialize returns
            box_body_sizes = self.controller.last_event.metadata["agent"]["fpinColliderSize"]
            scene["metadata"]["navMeshes"] = self.get_navmeshes(
                # collider scale is already baked in fpinColliderSize return
                box_body_sizes=box_body_sizes,
                min_side_as_radius=True,
                radius_epsilon=navmesh_radius_eps,
            )
            self.navmeshes = scene["metadata"]["navMeshes"]
            
         ###############################################

        try:
            self.step(action="CreateHouse", house=scene)
            # reset_event = self.controller.reset(scene=scene)
        except Exception as e:
            error_message = str(e)
            raise ValueError(error_message)
        

        self.set_object_filter([])
        

        self.room_poly_map, self.room_type_dict = get_rooms_polymap_and_type(
            self.current_scene_json
        )
        
            
        # Calibrate the agent
        self.calibrate_agent(copy.deepcopy(self.agent_params))
        
        
        # teleport_event = self.teleport_agent(**scene["metadata"]["agent"])

        teleport_event = self.step(
            action="RandomlyPlaceAgentOnNavMesh",
            n=200,  # Number of sampled points in Navmesh defaults to 200
        )
        

        # print("Camera Horizon after teleport: ", self.controller.last_event.metadata['agent']['cameraHorizon'])
        
        if not teleport_event.metadata["lastActionSuccess"]:

            raise Exception(
                f"FAILED TO TELEPORT AGENT TO the scene "
            )
        

        return reset_event

    def get_all_camera_parameters(self) -> Dict:
        """
        Returns a dictionary with the camera parameters for the navigation and manipulation cameras
        Each camera has position, rotation and field of view
        """
        navigation_camera_param = dict(
            position=self.controller.last_event.metadata["agentPositionRelativeCameraPosition"],
            rotation=self.controller.last_event.metadata["agentPositionRelativeCameraRotation"],
            fov=self.controller.last_event.metadata["fov"],
        )
        third_party_values = self.controller.last_event.metadata["thirdPartyCameras"][0]
        manipulation_camera_param = dict(
            position=third_party_values["agentPositionRelativeThirdPartyCameraPosition"],
            rotation=third_party_values["agentPositionRelativeThirdPartyCameraRotation"],
            fov=third_party_values["fieldOfView"],
        )
        return dict(
            navigation_camera=navigation_camera_param, manipulation_camera=manipulation_camera_param
        )

    # removed to induce errors for moving to new get_objects api
    # def get_all_objects_of_type(self, object_type):
    #     with self.include_object_metadata_context():
    #         return self.controller.last_event.objects_by_type(object_type)

    def get_visible_objects(
        self,
        which_camera: Literal["nav", "manip", "both"] = "nav",
        maximum_distance=2,
    ):
        # FYI: filtering by objects at this level has been removed to make best use
        # of the cache, but GetVisibleObjects still supports it with a list passed as objectIds=filter_object_ids.


        assert which_camera in ["nav", "manip", "both"]


        #TODO KE: I reached this issue! Not sure who implemented this functionality. currently in the middle of something. debug later

        if which_camera == 'nav':
            return self.controller.step(
                    "GetVisibleObjects",
                    maxDistance=maximum_distance,
                    # objectIds=filter_object_ids,
                ).metadata["actionReturn"]
        elif which_camera == 'manip':
            return self.controller.step(
                    "GetVisibleObjects",
                    maxDistance=maximum_distance,
                    thirdPartyCameraIndex=0,
                ).metadata["actionReturn"]
        else:
            return (
                self.controller.step(
                    "GetVisibleObjects",
                    maxDistance=maximum_distance,
                    # objectIds=filter_object_ids,
                ).metadata["actionReturn"]
                + self.controller.step(
                    "GetVisibleObjects",
                    maxDistance=maximum_distance,
                    thirdPartyCameraIndex=0,
                ).metadata["actionReturn"]
            )

    def get_approx_object_mask(
        self, object_id: str, which_camera: Literal["nav", "manip"], divisions: int
    ):
        step_dict = dict(
            action="GetApproxObjectMask",
            objectId=object_id,
            # thirdPartyCameraIndex=None if which_camera == "nav" else 0,
            divisions=divisions,
        )
        if which_camera == "manip":
            step_dict["thirdPartyCameraIndex"] = 0
        return self.step(**step_dict).metadata["actionReturn"]

    def object_is_visible_in_camera(
        self, object_id, which_camera: Literal["nav", "manip", "both"] = "nav", maximum_distance=2
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

    def get_locations_on_receptacle(self, receptacle_id):
        result = self.step(
            action="GetSpawnCoordinatesAboveReceptacle", objectId=receptacle_id, anywhere=True
        )
        return result.metadata["actionReturn"]

    def get_current_agent_position(self):
        return StretchState(self.controller).base_position
        # return self.controller.last_event.metadata["agent"]["position"]

    def get_current_agent_full_pose(self):
        return {
            **self.controller.last_event.metadata["agent"],
            "arm": self.controller.last_event.metadata["arm"],
        }

    def query_env(self, **kwargs):
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

        # print(self.controller.step(
        #     action="GetObjectMetadata", objectIds=[object_id], raise_for_failure=True
        # ))
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

    def get_agent_alignment_to_wall(self, wall_id, use_arm_orientation: bool = False):
        current_agent_pose = StretchState(self)
        wall_location = get_wall_center_floor_level(
            wall_id, y=current_agent_pose.base_position["y"]
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
            # TODO KIANA
            raise Exception(f"GetReachablePositions failed in {self.current_scene_json}")
            return []
        reachable_positions = rp_event.metadata["actionReturn"]
        return reachable_positions

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
            
            # TODO KE: THIS IS REAL HACKY
            self.list_of_last_action_success.append(event.metadata["lastActionSuccess"])
            num_failed_actions = self.list_of_last_action_success.count(False)
            if num_failed_actions >= 5:
                exception_error = (
                    f"Agent has consecutively failed {num_failed_actions} times"
                )
                if ALLOW_CONSECUTIVE_FAILURES:
                    warnings.warn(exception_error)
                else:
                    raise AgentIsStuckException(exception_error)
            

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
    def get_shortest_path_to_object_merged(
        self,
        object_id,
        initial_position=None,
        initial_rotation=None,
        specific_agent_meshes=None,
        attempt_path_improvement: bool = True,
        sample_from_navmesh=False,
        return_navmesh_id=False,
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
            initial_position.pop("theta")


        args = dict(
            action="GetShortestPath",
            objectId=object_id, 
            position=initial_position,
            navMeshIds=specific_agent_meshes,  # update to incorporate navmesh
            sampleFromNavmesh=sample_from_navmesh,
        )
        if initial_rotation is not None:
            args["rotation"] = initial_rotation
        event = self.step(**args)
        corners = None
        if event.metadata["lastActionSuccess"]:
            corners = event.metadata["actionReturn"]["corners"]
            if len(corners) != 0:
                self.last_successful_path = corners
                if attempt_path_improvement and len(corners) > 2:
                    corners = snap_to_skeleton(
                        controller=self,
                        corners=corners,
                    )
        else:
            print(args)
        if not return_navmesh_id:
            return corners
        else:
            return (corners, None)


    def get_shortest_path_to_object(
        self,
        object_id,
        initial_position=None,
        initial_rotation=None,
        specific_agent_meshes=None,
        attempt_path_improvement: bool = True,
        sample_from_navmesh=False,
        return_navmesh_id=False,
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
            initial_position.pop("theta")
            # initial_position = self.controller.last_event.metadata["agent"]["position"]

        rotations = [dict(x=0.0, y=0.0, z=0.0), dict(x=0.0, y=90.0, z=0.0), dict(x=0.0, y=180.0, z=0.0), dict(x=0.0, y=270.0, z=0.0)]
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

                # try:
                #     # Apparently we don't really need this
                #     # First_half_replan can be None in the split_and_replan_paths function
                #     # TODO Ainaz: Not sure why this is happening
                #     if (
                #         nav_mesh_id > 1
                #         and not mesh_restriction
                #         and attempt_path_improvement
                #         and len(corners) > 4
                #     ):
                #         corners = self.split_and_replan_paths(
                #             initial_position, corners[-1], corners, recursion_depth=1
                #         )
                # except:
                #     pass
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
        test_all_navmeshes=False,
        sample_from_navmesh=False,
        return_navmesh_id=False,
    ) -> bool:
        """
        Checks if a shortest path to an object from an initial position exists. This is faster than
        `get_shortest_path_to_object` as we will only use the most general nav mesh.

        :param object_id: string with id of the object
        :param initial_position: dict(x=float, y=float, z=float) with the desired initial rotation
        :param initial_rotation: dict(x=float, y=float, z=float) representing rotation around axes or None
        :return:
        """
        result = self.get_shortest_path_to_object(
            object_id=object_id,
            initial_position=initial_position,
            initial_rotation=initial_rotation,
            specific_agent_meshes=[self.agent_ids[-1]] if not test_all_navmeshes else None,
            attempt_path_improvement=False,
            return_navmesh_id=return_navmesh_id,
            sample_from_navmesh=sample_from_navmesh,
        )
        return (result is not None) if not return_navmesh_id else (result[0] is not None, result[1])

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
        sample_from_navmesh=False,
        return_navmesh_id=False,
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
            initial_position.pop("theta")
            # initial_position = self.controller.last_event.metadata["agent"]["position"]

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
        if not return_navmesh_id:
            return corners
        else:
            return (corners, None)

    def num_pixels_visible(self, object_id, manipulation_camera=False):
        assert (
            "renderInstanceSegmentation" in self.initialization_args
            and self.initialization_args["renderInstanceSegmentation"]
        )
        if manipulation_camera:
            masks = self.manipulation_camera_segmentation
        else:
            masks = self.navigation_camera_segmentation

        if object_id not in masks:
            return 0

        mask = masks[object_id]
        return mask.sum()

    def is_object_visible_enough_for_interaction(self, object_id: str, manipulation_camera=True):
        return is_any_object_sufficiently_visible_and_in_center_frame(
            controller=self,
            object_ids=[object_id],
            manipulation_camera=manipulation_camera,
            object_synset=None,
        )

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


class FPINStochasticController(FPINController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_rand_action_kwargs = None

    def step(self, **kwargs):

        if "action" in kwargs and kwargs["action"] in ["MoveAhead", "RotateAgent"]:
            rand = np.random.normal(0, 1, 1)[0]

            # TODO Add stochastic motion for arm
            if "action" in kwargs and kwargs["action"] == "MoveAgent":
                kwargs["ahead"] += 0.01 * rand
            if "action" in kwargs and kwargs["action"] == "RotateAgent":
                kwargs["degrees"] += 0.5 * rand
            self.last_rand_action_kwargs = kwargs
        return super(FPINStochasticController, self).step(**kwargs)
