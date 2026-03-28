import copy
import json
import math
import random
from typing import Dict
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import os

from utils.constants.objaverse_data_dirs import OBJAVERSE_ASSETS_DIR, OBJAVERSE_BODY_ASSETS_DIR
from utils.constants.FPIN_utils import DEFAULT_PARAMS, AGENT_BODY_PARAMS



# def load_fpin_asset_and_sizes():
#     with open("scripts/fpin_body_sizes.json") as f:
#         data = json.load(f)
#         return data
    
# FPIN_ASSETS_AND_SIZES = load_fpin_asset_and_sizes()
# FPIN_ASSETS_AND_SIZES = {
#     k: v
#     for k, v in FPIN_ASSETS_AND_SIZES.items()
#     if v["fpin_collider_body_size"]["x"] < 0.5 and v["fpin_collider_body_size"]["z"] < 0.5
# }
# LIST_OF_POSSIBLE_FPIN_ASSETS = list(
#     FPIN_ASSETS_AND_SIZES.keys()
# )  # TODO KIANA THIS ONLY INCLUDES VERY SMALL OBJECTS 130/2300

def get_vertical_fov(fov_h, w, h):
        return int(math.degrees(2 * math.atan(math.tan(math.radians(fov_h) / 2) * h / w)))
    
def get_horizontal_fov(fov_v, w, h):
    return int(math.degrees(2 * math.atan(math.tan(math.radians(fov_v) / 2) * w / h)))


def get_nav_mesh_config_from_box(
    base_collider_scale,
    nav_mesh_scale,
    box_body_sizes,
    nav_mesh_id,
    min_side_as_radius=False,
    radius_epsilon=0.0,
):
    # This can be changed to fit specific needs
    capsuleHeight = box_body_sizes["y"] * base_collider_scale["y"] * nav_mesh_scale

    # This is now correct if passed `fpinColliderSize`, from initialize or agent metadata as its the full x,z sides
    # also base_collider_scale is already included in `fpinColliderSize` so possibly remove this
    halfBoxX = (box_body_sizes["x"] / 2.0) * base_collider_scale["x"] * nav_mesh_scale
    halfBoxZ = (box_body_sizes["z"] / 2.0) * base_collider_scale["z"] * nav_mesh_scale


    capsuleOverEstimateRadius = math.sqrt(halfBoxX * halfBoxX + halfBoxZ * halfBoxZ)
    capsuleUnderEstimateRadius = min(halfBoxX, halfBoxZ)

    return {
        "id": nav_mesh_id,
        "agentRadius": (
            capsuleUnderEstimateRadius if min_side_as_radius else capsuleOverEstimateRadius
        )
        + radius_epsilon,
        "agentHeight": capsuleHeight,
    }


@dataclass
class AgentParams:
    print(DEFAULT_PARAMS)
    first_camera_fov: int = copy.deepcopy(DEFAULT_PARAMS["first_camera_fov"])
    second_camera_fov: int = copy.deepcopy(DEFAULT_PARAMS["second_camera_fov"])

    width: int = copy.deepcopy(DEFAULT_PARAMS["width"])
    height: int = copy.deepcopy(DEFAULT_PARAMS["height"])

    first_camera_position: Dict[str, float] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_PARAMS["first_camera_position"])
    )
    first_camera_rotation: Dict[str, float] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_PARAMS["first_camera_rotation"])
    )
    second_camera_position: Dict[str, float] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_PARAMS["second_camera_position"])
    )
    second_camera_rotation: Dict[str, float] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_PARAMS["second_camera_rotation"])
    )
    agent_body_params: Dict[str, float] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_PARAMS["agent_body_params"])
    )

    @staticmethod
    def get_body_size_after_load(agent_asset):
        # actual size of body with scale (1,1,1)
        if agent_asset == 'ghost':
            return dict(x=1.0, y=1.0, z=1.0)
        elif agent_asset == 'locobot':
            return dict(x=0.351806640625, y=0.889404296875, z=0.4008026123046875)
        elif agent_asset == 'stretch':
            # return dict(x=0.3900543212890625, y=1.415557861328125, z=0.32861328125) # with 0.7 scale, closer to the real size
            return dict(x=0.557220458984375, y=1.415557861328125, z=0.32861328125) # actual strech asset size (bigger than real size)
        elif agent_asset == 'unitree_a1':
            return dict(x=0.30529022216796875, y=0.33652496337890625, z=0.638214111328125) # actual asset size (bigger than real size)
        # elif agent_asset == 'objaverse':
        #     return FPIN_ASSETS_AND_SIZES[self.get_body_id()]["fpin_collider_body_size"] 
        else:
            raise ValueError(f"Unknown agent asset: {agent_asset}")

    def get_collider_scale(self):
        return self.agent_body_params["colliderScaleRatio"]


    def get_body_id(self, agent_asset):
        if agent_asset == 'locobot':
            return "LocoBotSimObj"
        elif agent_asset == 'stretch':
            return "StretchBotSimObj"
        elif agent_asset == 'unitree_a1':
            return "unitree_a1"
        elif agent_asset == 'ghost':
            return "ghost"
        elif agent_asset == 'objaverse':
            return self.agent_body_params["bodyAsset"]["dynamicAsset"]["id"]

    def to_dict(self) -> dict:
        result = {
            "first_camera_fov": self.first_camera_fov,
            "second_camera_fov": self.second_camera_fov,
            "first_camera_rotation": self.first_camera_rotation,
            "second_camera_rotation": self.second_camera_rotation,
            "first_camera_position": self.first_camera_position,
            "second_camera_position": self.second_camera_position,
            "width": self.width,
            "height": self.height,
            "agent_body_params": self.agent_body_params,
        }
        return copy.deepcopy(result)

    def tostring(self) -> str:
        dict_to_print = self.to_dict()
        dict_to_print["agent_body_params"] = "..."
        return dict_to_short_string(dict_to_print)

    @staticmethod
    def from_dict(data: dict):
        return AgentParams(**data)


def dict_to_short_string(d):
    # Helper function to abbreviate keys
    def abbrev_key(key):
        abbreviations = {
            "first_camera_fov": "fc_fov",
            "second_camera_fov": "sc_fov",
            "first_camera_rotation": "fc_rot",
            "second_camera_rotation": "sc_rot",
            "first_camera_position": "fc_pos",
            "second_camera_position": "sc_pos",
            "width": "w",
            "height": "h",
            "agent_body_params": "abp",
        }
        return abbreviations.get(key, key)

    # Helper function to simplify values
    def simplify_value(value):
        if isinstance(value, dict):
            return {
                abbrev_key(k): round(v, 2) if isinstance(v, float) else v for k, v in value.items()
            }
        elif isinstance(value, float):
            return round(value, 2)
        return value
        # Simplify dictionary

    simplified_dict = {abbrev_key(k): simplify_value(v) for k, v in d.items()}

    # Convert to string (e.g., JSON)
    import json

    short_string = json.dumps(simplified_dict)

    return short_string


def round_to_even(n):
    # Subtract the remainder of dividing n by 2, then add 2 if the remainder was 1 or more (rounding up to the next even)
    return n - (n % 2) + (2 if n % 2 else 0)


class AgentParamRandomizer:
    def __init__(self, parameters_to_randomize=[], exact_values: dict = None, agent_asset=None):
        self.sub_parameters = [
            "first_camera_position_x",
            "first_camera_position_y",
            "first_camera_position_z",
            "second_camera_position_x",
            "second_camera_position_y",
            "second_camera_position_z",
        ]
        ############################
        # Agent Asset
        ############################
        self.agent_asset = agent_asset
        print(f"Agent Asset: {self.agent_asset}")
        DEFAULT_PARAMS["agent_body_params"] = AGENT_BODY_PARAMS[self.agent_asset]

        # The format of this dictionary is:
        # {asset_id: {"size": {"x": x, "y": y, "z": z}, "fpin_collider_body_size": {"x": x, "y": y, "z": z}}    }
        if agent_asset == 'ghost':
            self.possible_agent_body_ids = ["ghost"]
        # else:
        #     self.possible_agent_body_ids = LIST_OF_POSSIBLE_FPIN_ASSETS



        if parameters_to_randomize == "ALL":
            parameters_to_randomize = list(AgentParams.__annotations__.keys()) + self.sub_parameters
        assert set(parameters_to_randomize).issubset(
            set(list(AgentParams.__annotations__.keys()) + self.sub_parameters)
        )
        self.deterministic = False
        if exact_values is not None:
            self.exact_values = exact_values
            self.deterministic = True

        if exact_values is not None:
            print("Initializing AgentParamRandomizer with exact_values", exact_values)
            assert len(parameters_to_randomize) == 0

        elif len(parameters_to_randomize) == 0:
            print("Initializing AgentParamRandomizer with no parameters to randomize")
        else:
            print("Initializign AgentParamRandomizer with", parameters_to_randomize)


        self.parameters_to_randomize = parameters_to_randomize

        # parameter ranges
        self.FOV_RANGE = [40, 100]
        self.PITCH_RANGE_NAV = [-20, 40] # rotatation_x
        self.PITCH_RANGE_MANIP = [-20, 60] # rotatation_x
        self.YAW_RANGE = [0, 360] # rotation_y
        self.WIDTH_RANGE = self.HEIGHT_RANGE = [112, 448]
        self.CAMERA_HEIGHT_RANGE = [0.3, 1.5]
        self.CAMERA_X_Z_RANGE = [-0.3, 0.3]
        
        if agent_asset == 'ghost':
            self.COLLIDER_ABSOLUTE_SCALE_RANGE_X_Z = [0.2, 0.5]

            self.COLLIDER_ABSOLUTE_SCALE_RANGE_Y = [0.3, 1.5]


        



    def get_random_value(self, range):

        if type(range[0]) == int and type(range[1]) == int:
            return np.random.randint(range[0], range[1])
        elif type(range[0]) == float and type(range[1]) == float:
            return np.random.uniform(range[0], range[1])
        else:
            print("Types don not match", range[0], range[1])
            raise ValueError("Range must be either int or float")
        
    

    def get_random_agent_param(self):

        if self.deterministic:
            assert self.exact_values is not None
            agent_params = self.get_exact_agent_param(self.exact_values)
            agent_params.agent_body_params = AGENT_BODY_PARAMS[self.agent_asset]
            return agent_params

        if len(self.parameters_to_randomize) == 0:
            return AgentParams()

        agent_body_params = copy.deepcopy(AGENT_BODY_PARAMS[self.agent_asset])

        ####################################
        # Randomize collider size
        ####################################
        if "agent_body_params" in self.parameters_to_randomize:
            if self.agent_asset == 'objaverse':
                agent_type = random.choice(self.possible_agent_body_ids)
                agent_body_params["bodyAsset"]["dynamicAsset"]["id"] = agent_type
                agent_body_params["colliderScaleRatio"]["x"] = self.get_random_value(
                    self.COLLIDER_RELATIVE_SCALE_RANGE_X_Y_Z
                )
                agent_body_params["colliderScaleRatio"]["y"] = self.get_random_value(
                    self.COLLIDER_RELATIVE_SCALE_RANGE_X_Y_Z
                )
                agent_body_params["colliderScaleRatio"]["z"] = self.get_random_value(
                    self.COLLIDER_RELATIVE_SCALE_RANGE_X_Y_Z
                )
            else:
                agent_body_params["colliderScaleRatio"]["x"] = self.get_random_value(
                    self.COLLIDER_ABSOLUTE_SCALE_RANGE_X_Z
                )
                agent_body_params["colliderScaleRatio"]["y"] = self.get_random_value(
                    self.COLLIDER_ABSOLUTE_SCALE_RANGE_Y
                )
                agent_body_params["colliderScaleRatio"]["z"] = self.get_random_value(
                    self.COLLIDER_ABSOLUTE_SCALE_RANGE_X_Z
                )
        # collider_size = {}

        

        ####################################
        # Randomize camera position and offset
        ####################################
        camera_x_range = self.CAMERA_X_Z_RANGE
        camera_z_range = self.CAMERA_X_Z_RANGE

        if self.agent_asset != 'objaverse':  # To make sure camera is within the collider
            # TODO KIANA THIS NEEDS TO BE IMPLEMENTED FOR OBJAVERSE AGENTS TOO
            # get the original body size with scale (1,1,1)
            body_size = AgentParams.get_body_size_after_load(self.agent_asset)
            # correct size with the scale
            agent_dim_z = body_size["z"] * agent_body_params["colliderScaleRatio"]["z"]
            agent_dim_x = body_size["x"] * agent_body_params["colliderScaleRatio"]["x"]
            agent_dim_y = body_size["y"] * agent_body_params["colliderScaleRatio"]["y"]

            camera_x_range = [-agent_dim_x / 2.5, agent_dim_x / 2.5]
            camera_z_range = [-agent_dim_z / 2.5, agent_dim_z / 2.5]
            camera_y_range = [self.CAMERA_HEIGHT_RANGE[0], agent_dim_y]
            
            offset_x_range = [-agent_dim_x / 2.5, agent_dim_x / 2.5]
            offset_z_range = [-agent_dim_z / 2.5, agent_dim_z / 2.5]

        first_camera_position = copy.deepcopy(DEFAULT_PARAMS["first_camera_position"])
        second_camera_position = copy.deepcopy(DEFAULT_PARAMS["second_camera_position"])
        first_camera_position["y"] = self.get_random_value(camera_y_range)
        first_camera_position["z"] = self.get_random_value(camera_z_range)
        first_camera_position["x"] = self.get_random_value(camera_x_range)
        second_camera_position["y"] = self.get_random_value(camera_y_range)
        second_camera_position["z"] = self.get_random_value(camera_z_range)
        second_camera_position["x"] = self.get_random_value(camera_x_range)


        ####################################
        # Randomize camera rotation
        ####################################
        first_camera_rotation = copy.deepcopy(DEFAULT_PARAMS["first_camera_rotation"])
        second_camera_rotation = copy.deepcopy(DEFAULT_PARAMS["second_camera_rotation"])
        pitch_range_manip = self.PITCH_RANGE_MANIP
        pitch_range_nav = self.PITCH_RANGE_NAV
        if second_camera_position["y"] < 0.5:
            pitch_range_manip[-1] = 40
        first_camera_rotation["x"] = self.get_random_value(pitch_range_nav)
        second_camera_rotation["x"] = self.get_random_value(pitch_range_manip)
        # first camera always looks forward, we randomize the second camera
        second_camera_rotation["y"] = self.get_random_value(self.YAW_RANGE)
        
        
        ####################################
        # adjust stretch camera position based on the camera yaw
        ####################################
        # if using stretch asset, make sure position_z>0 for first camera and position_x>0 for second camera. Otherwise, agent body will be in the camera view
        if self.agent_asset == 'stretch':
            first_camera_position["z"] = self.get_random_value([0.0, agent_dim_z / 2])
            # second camera position is based on the camera yaw
            if second_camera_rotation["y"] >=0  and second_camera_rotation["y"] < 90:
                camera_x_range = [-agent_dim_x / 2, 0.0]
                camera_z_range = [0.0, agent_dim_z / 2]
            elif second_camera_rotation["y"] >= 90 and second_camera_rotation["y"] < 180:
                camera_x_range = [-agent_dim_x / 2, 0.0]
                camera_z_range = [-agent_dim_z / 2, 0.0]
            elif second_camera_rotation["y"] >= 180 and second_camera_rotation["y"] < 270:
                camera_x_range = [0.0, agent_dim_x / 2]
                camera_z_range = [-agent_dim_z / 2, 0.0]
            else:
                camera_x_range = [0.0, agent_dim_x / 2]
                camera_z_range = [0.0, agent_dim_z / 2]
            second_camera_position["x"] = self.get_random_value(camera_x_range)
            second_camera_position["z"] = self.get_random_value(camera_z_range)
        
        
        ####################################
        # Randomize origin offset
        ####################################
        if "agent_body_params" in self.parameters_to_randomize:
            agent_body_params['originOffsetX'] = self.get_random_value(offset_x_range)
            agent_body_params['originOffsetZ'] = self.get_random_value(offset_z_range)

        ####################################
        # Randomize H, W
        ####################################
        width = 224
        height = self.get_random_value(self.HEIGHT_RANGE)
        if random.uniform(0, 1) < 0.5:
            # Swap width and height
            width, height = height, width
            
        # Apparently this is necessary for ffmpeg
        width = round_to_even(width)
        height = round_to_even(height)
        print(f"width: {width}, height: {height}")

        ####################################
        # Randomize camera FOV
        ####################################
        if height < width:
            max_fov_y = get_vertical_fov(fov_h=120, w=width, h=height)
            print(f"max_fov_y: {max_fov_y}")
            NEW_FOV_RANGE = [self.FOV_RANGE[0], max_fov_y] # to make sure horizontal fov is less than 120
            
        else:
            min_fov_y = get_vertical_fov(fov_h=40, w=width, h=height)
            print(f"min_fov_y: {min_fov_y}")
            NEW_FOV_RANGE = [min_fov_y, self.FOV_RANGE[1]] # to make sure horizontal fov is more than 40

        first_camera_fov = self.get_random_value(NEW_FOV_RANGE)
        second_camera_fov = self.get_random_value(NEW_FOV_RANGE)


        dict_to_set = {}
        for param in self.parameters_to_randomize:
            assert eval(param) is not None, f"Parameter {param} is not defined"
            dict_to_set[param] = eval(param)



        return AgentParams.from_dict(dict_to_set)

    def get_exact_agent_param(self, agent_params: dict):
        return AgentParams.from_dict(agent_params)


def two_assets_are_similar(asset1, asset2):
    asset_id_2 = None
    asset_id_1 = None
    try:
        asset_id_1 = asset1["bodyAsset"]["dynamicAsset"]["id"]
    except Exception:
        pass
    try:
        asset_id_2 = asset2["bodyAsset"]["dynamicAsset"]["id"]
    except Exception:
        pass
    return asset_id_1 is None or asset_id_2 is None or asset_id_1 == asset_id_2


if __name__ == "__main__":
    import pdb

    agent_randomizer = AgentParamRandomizer()
    for i in range(10):
        agent_params = agent_randomizer.get_random_agent_param()
        print("with AgentParamRandomizer()", agent_params.to_dict())

    agent_randomizer = AgentParamRandomizer(
        exact_values=dict(first_camera_fov=69, width=224, height=396, first_camera_rotation=30)
    )

    for i in range(10):
        agent_params = agent_randomizer.get_random_agent_param()
        print(
            "with AgentParamRandomizer(AgentParamRandomizer(exact_values=dict(first_camera_fov=42, width=224, height=396, first_camera_rotation=0)))",
        )
        print(agent_params.to_dict())

    agent_randomizer = AgentParamRandomizer(parameters_to_randomize="ALL")

    for i in range(10):
        agent_params = agent_randomizer.get_random_agent_param()
        print(
            "with AgentParamRandomizer(ALL)",
        )
        print(agent_params.to_dict())


    pdb.set_trace()
