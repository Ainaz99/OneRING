
import os

from utils.constants.objaverse_data_dirs import OBJAVERSE_BODY_ASSETS_DIR, OBJAVERSE_DATASETS_DIR

# true for evaluation/RL, false for data generation
ALLOW_CONSECUTIVE_FAILURES = True
VALIDATE_BENCHMARK = False

PADDING = False
AGENT_MODE = "fpin"  #stretch or "fpin"
CUTOFF = 6
MODE = 'RL'  # 'RL' or 'DATAGEN' or 'EVAL'

DEFAULT_ASSET = os.getenv('DEFAULT_ASSET', 'ghost') # "locobot"/"stretch"/"objaverse"/"ghost"/"unitree_a1"

# PARAMS_TO_RANDOMIZE = []
PARAMS_TO_RANDOMIZE = [
    'first_camera_position', 
    'first_camera_rotation', 
    'first_camera_fov', 
    'second_camera_position', 
    'second_camera_rotation', 
    'second_camera_fov', 
    'width', 
    'height',
    'agent_body_params',
]



FIRST_CAMERA_POSITION = {}
SECOND_CAMERA_POSITION = {}
FIRST_CAMERA_ROTATION = {}
SECOND_CAMERA_ROTATION = {}
FIRST_CAMERA_FOV = {}
SECOND_CAMERA_FOV = {}
WIDTH = {}
HEIGHT = {}
AGENT_BODY_PARAMS = {}

# stretch
FIRST_CAMERA_POSITION['stretch'] = {"x": 0.00192035, "y": +0.9009926-0.000999797135591507+0.5447009, "z": 0.0678804}
SECOND_CAMERA_POSITION['stretch'] = {"x": 0.05390513, "y": 0.9009926-0.000999797135591507+0.5238336, "z": -0.05884857}
FIRST_CAMERA_ROTATION['stretch'] = {"x": 27, "y": 0.0, "z": 0.0}
SECOND_CAMERA_ROTATION['stretch'] = {"x": 33, "y": 90.0, "z": 0.0}
FIRST_CAMERA_FOV['stretch'] = 59
SECOND_CAMERA_FOV['stretch'] = 59
WIDTH['stretch'] = 396
HEIGHT['stretch'] = 224
AGENT_BODY_PARAMS['stretch'] = dict(
    bodyAsset={"assetId": "StretchBotSimObj"},
    originOffsetX=0.0,
    originOffsetZ=0.1157837,
    colliderScaleRatio={"x": 0.7, "y": 1, "z": 1},
    useAbsoluteSize=False,
    useVisibleColliderBase=False,
)


# locobot
FIRST_CAMERA_POSITION['locobot'] = {"x": 0.0, "y": +0.9009997844696045-0.000999797135591507-0.0312, "z": 0.0}
SECOND_CAMERA_POSITION['locobot'] = SECOND_CAMERA_POSITION['stretch'] # doesn't have second camera
FIRST_CAMERA_ROTATION['locobot'] = {"x": 0.0, "y": 0.0, "z": 0.0}
SECOND_CAMERA_ROTATION['locobot'] = SECOND_CAMERA_ROTATION['stretch'] # doesn't have second camera
FIRST_CAMERA_FOV['locobot'] = 42 #60
SECOND_CAMERA_FOV['locobot'] = 42 #60
WIDTH['locobot'] = 396 #400
HEIGHT['locobot'] = 224 #300
AGENT_BODY_PARAMS['locobot'] = dict(
    bodyAsset={"assetId": "LocoBotSimObj"},
    originOffsetX=0.0,
    originOffsetZ=-0.0,
    colliderScaleRatio={"x": 1, "y": 1, "z": 1},
    useAbsoluteSize=False,
    useVisibleColliderBase=False,
)


# unitree_a1
# FIRST_CAMERA_POSITION['unitree_a1'] = {"x": 0.0364 * 0.34, "y": 0.4304 * 0.34, "z": 0.9597 * 0.34}
FIRST_CAMERA_POSITION['unitree_a1'] = {"x": 0.0091, "y": 0.2998, "z": 0.2723}
SECOND_CAMERA_POSITION['unitree_a1'] = SECOND_CAMERA_POSITION['stretch'] # doesn't have second camera
FIRST_CAMERA_ROTATION['unitree_a1'] = {"x": 0.0, "y": 0.0, "z": 0.0}
SECOND_CAMERA_ROTATION['unitree_a1'] = SECOND_CAMERA_ROTATION['stretch'] # doesn't have second camera
FIRST_CAMERA_FOV['unitree_a1'] = 42
SECOND_CAMERA_FOV['unitree_a1'] = 42
WIDTH['unitree_a1'] = 480
HEIGHT['unitree_a1'] = 270
AGENT_BODY_PARAMS['unitree_a1'] = dict(
    bodyAsset=dict(
        dynamicAsset={
            "id": "unitree_a1",
            "dir": os.path.abspath(
                f"{os.getenv('OBJAVERSE_DATA_DIR')}/2024_03_01/assets"
                # "/data/output/khzeng/objaverse_assets/2023_07_28_msgpack/2024_03_01/assets"
            ), 
        }
    ),
    originOffsetX=0.0,
    originOffsetZ=0.042,
    colliderScaleRatio={"x": 1.0, "y": 1.0, "z": 1.0},
    useAbsoluteSize=False,
    useVisibleColliderBase=False,
)

# ghost
FIRST_CAMERA_POSITION['ghost'] = FIRST_CAMERA_POSITION['stretch']
SECOND_CAMERA_POSITION['ghost'] = SECOND_CAMERA_POSITION['stretch'] 
FIRST_CAMERA_ROTATION['ghost'] = FIRST_CAMERA_ROTATION['stretch']
SECOND_CAMERA_ROTATION['ghost'] = SECOND_CAMERA_ROTATION['stretch'] 
FIRST_CAMERA_FOV['ghost'] = FIRST_CAMERA_FOV['stretch']
SECOND_CAMERA_FOV['ghost'] = SECOND_CAMERA_FOV['stretch']
WIDTH['ghost'] = WIDTH['stretch']
HEIGHT['ghost'] = HEIGHT['stretch']
AGENT_BODY_PARAMS['ghost'] = {
    "originOffsetX": 0.0,  # TODO KIANA from winson: and remember, don't set the origin offset beyond the absolute size. you can but I don't recommend it unless the agent is a boomerang where the center of rotation is expected to be offset haha
    "originOffsetZ": 0.0,  # TODO KIANA these three need to be randomized as well
    "colliderScaleRatio": {"x": 0.3, "y":0.5, "z": 0.3},
    "useAbsoluteSize": True,
    "useVisibleColliderBase": True,
}


# DEFAULT_AGENT_BODY_PARAMS = dict(
#     # bodyAsset= {"assetId": "Toaster_5"},
#     bodyAsset=dict(
#         dynamicAsset={
#             "id": "2ac18246873f4644af4f9f48361fc599",
#             "dir": os.path.abspath(
#                 OBJAVERSE_BODY_ASSETS_DIR
#                 # "/Users/kianae/Desktop/new_objaverse/2024_03_01/assets" // this magic path breaks things
#             ),  # TODO_LATER this path needs to be set somehow. download new objects?
#         }
#     ),
#     originOffsetX=0.0,
#     originOffsetZ=0.0,
#     colliderScaleRatio={"x": 1, "y": 1, "z": 1},
#     useAbsoluteSize=False,
#     useVisibleColliderBase=True,
# )


DEFAULT_PARAMS = {
            "first_camera_position": FIRST_CAMERA_POSITION[DEFAULT_ASSET],
            "first_camera_rotation": FIRST_CAMERA_ROTATION[DEFAULT_ASSET],
            "second_camera_position": SECOND_CAMERA_POSITION[DEFAULT_ASSET],
            "second_camera_rotation": SECOND_CAMERA_ROTATION[DEFAULT_ASSET],
            "first_camera_fov": FIRST_CAMERA_FOV[DEFAULT_ASSET],
            "second_camera_fov": SECOND_CAMERA_FOV[DEFAULT_ASSET],
            "agent_body_params": AGENT_BODY_PARAMS[DEFAULT_ASSET],
            "height": HEIGHT[DEFAULT_ASSET],
            "width": WIDTH[DEFAULT_ASSET],
        }


BODY_PARAMS = {}

BODY_PARAMS['stretch']= dict(
    originOffsetX=0.0,
    originOffsetZ=0.1157837,
    colliderScaleRatio={"x": 0.3900543212890625, "y": 1.415557861328125, "z": 0.32861328125},
    useAbsoluteSize=True,
    useVisibleColliderBase=False,
)

BODY_PARAMS['locobot']= dict(
    originOffsetX=0.0,
    originOffsetZ=-0.0,
    colliderScaleRatio={"x": 0.351806640625, "y": 0.889404296875, "z": 0.4008026123046875},
    useAbsoluteSize=True,
    useVisibleColliderBase=False,
)

BODY_PARAMS['unitree_a1']= dict(
    originOffsetX=0.0,
    originOffsetZ=0.042,
    colliderScaleRatio={"x": 0.30529022216796875, "y": 0.33652496337890625, "z": 0.638214111328125},
    useAbsoluteSize=True,
    useVisibleColliderBase=False,
)








