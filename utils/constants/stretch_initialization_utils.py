import os

import os

import ai2thor.fifo_server
from torch.distributions.utils import lazy_property

from torch.distributions.utils import lazy_property

from utils.constants.objaverse_data_dirs import OBJAVERSE_ASSETS_DIR

from utils.constants.FPIN_utils import AGENT_MODE, MODE


MODEL_43_WIDTH, MODEL_43_HEIGHT = 320, 240



STRETCH_COMMIT_ID = "7c61ad0c3b4a27f169ee02dab01bb83b6303696f" if AGENT_MODE == "stretch" else "d3370a1dabb80fde5793c92c9e178c80aa91d801"


try:
    from ai2thor.hooks.procedural_asset_hook import (
        ProceduralAssetHookRunner,
        get_all_asset_ids_recursively,
        create_assets_if_not_exist,
    )
except ImportError:
    raise ImportError(
        "Cannot import `ProceduralAssetHookRunner`. Please install the appropriate version of ai2thor:\n"
        f"```\npip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+d3370a1dabb80fde5793c92c9e178c80aa91d801\n```"
    )

AGENT_ROTATION_DEG = 30
AGENT_MOVEMENT_CONSTANT = 0.2
HORIZON = 0  # RH: Do not change from 0! this is now set elsewhere with RotateCameraMount actions
GRID_SIZE = 0.2
ARM_MOVE_CONSTANT = 0.1
WRIST_ROTATION = 10

EMPTY_BBOX = [1000, 1000, 1000, 1000, 0]
EMPTY_DOUBLE_BBOX = EMPTY_BBOX + EMPTY_BBOX

ORIGINAL_INTEL_W, ORIGINAL_INTEL_H = 1280, 720
INTEL_CAMERA_WIDTH, INTEL_CAMERA_HEIGHT = 396, 224


INTEL_WIDTH_CROPPED, INTEL_HEIGHT_CROPPED = 384, 224
INTEL_VERTICAL_FOV = 59

AGENT_RADIUS_LIST = [(0, 0.5), (1, 0.4), (2, 0.3), (3, 0.2)]


# Smaller to larger so that there is a higher chance to find path first, maybe do a sort in actual sp function
# AGENT_COLLIDER_NAVMESH_REL_SCALE = [(0, 1), (1, 1.1), (2, 1.3), (3, 1.5)]
AGENT_COLLIDER_NAVMESH_REL_SCALE = [(0, 1.5), (1, 1.3), (2, 1.1), (3, 1)]

# AGENT_RADIUS_LIST = [(0, 0.2)]
# AGENT_COLLIDER_NAVMESH_REL_SCALE = [(0, 1)]


# AGENT_RADIUS_LIST = [(0, 0.3), (1, 0.2)]
# AGENT_COLLIDER_NAVMESH_REL_SCALE = [(0, 1.1), (1, 1)]

# AGENT_RADIUS_LIST = [(0, 1)]
# AGENT_COLLIDER_NAVMESH_REL_SCALE = [(0, 1)]

MAXIMUM_SERVER_TIMEOUT = 1000  # default : 100 Need to increase this for cloudrendering

AGENTS_BASE_HEIGHT = 0.900992214679718 #0.000999797135591507 #0.900992214679718

def assert_agent_objaverse_mesh_exists(agent_params):
    if (
        "bodyAsset" in agent_params
        and "dynamicAsset" in agent_params["bodyAsset"]
        and "id" in agent_params["bodyAsset"]["dynamicAsset"]
    ):
        dir = agent_params["bodyAsset"]["dynamicAsset"]["dir"]
        id = agent_params["bodyAsset"]["dynamicAsset"]["id"]
        assert os.path.exists(os.path.join(dir, id)), f"Asset does not exist {id} at path {dir}."


class ProceduralAssetHookRunnerResetOnNewHouse(ProceduralAssetHookRunner):
    @lazy_property
    def last_asset_id_set(self):
        return set()
    @lazy_property
    def last_asset_id_set(self):
        return set()

    def Initialize(self, action, controller):
        if self.asset_limit > 0:
            return controller.step(
                action="DeleteLRUFromProceduralCache", assetLimit=self.asset_limit
            )

    def CreateHouse(self, action, controller):
        house = action["house"]
        asset_ids = get_all_asset_ids_recursively(house["objects"], [])
        asset_ids_set = set(asset_ids)
        if not asset_ids_set.issubset(self.last_asset_id_set):
            controller.step(action="DeleteLRUFromProceduralCache", assetLimit=0)
            self.last_asset_id_set = set(asset_ids)

        if STRETCH_COMMIT_ID == "5e43486351ac6339c399c199e601c9dd18daecc3":
            # This is for the old THOR, the one we used for CVPR submission
            return create_assets_if_not_exist(
                controller=controller,
                asset_ids=asset_ids,
                asset_directory=self.asset_directory,
                asset_symlink=self.asset_symlink,
                stop_if_fail=self.stop_if_fail,
            )
        else:
            return create_assets_if_not_exist(
                controller=controller,
                asset_ids=asset_ids,
                asset_directory=self.asset_directory,
                asset_symlink=self.asset_symlink,
                stop_if_fail=self.stop_if_fail,
                copy_to_dir=os.path.join(controller._build.base_dir, self.target_dir),
                load_file_in_unity=True,  # Turn on the load_file_in_unity to improve loading speed
            )


_ACTION_HOOK_RUNNER = ProceduralAssetHookRunnerResetOnNewHouse(
    asset_directory=OBJAVERSE_ASSETS_DIR,
    asset_symlink=True,
    verbose=True,
    asset_limit=200,
    load_file_in_unity=True,
)

# _ACTION_HOOK_RUNNER = ProceduralAssetHookRunner(
#     asset_directory=OBJAVERSE_ASSETS_DIR, asset_symlink=True, verbose=True, asset_limit=200
# )

PHYSICS_SETTLING_TIME = 1.0

MAXIMUM_DISTANCE_ARM_FROM_AGENT_CENTER = (
    1.0
    # 0.8673349051766235  # Computed with fixed arm agent, should have pairity with real
)

if os.getenv("SAVE_DEPTH") is not None:
    SAVE_DEPTH = bool(os.getenv("SAVE_DEPTH"))
elif os.getenv("RENDER_DEPTH") is not None:
    SAVE_DEPTH = bool(os.getenv("RENDER_DEPTH"))
else:
    SAVE_DEPTH = False

STRETCH_ENV_ARGS = dict(
    gridSize=GRID_SIZE
    * 0.75,  # Intentionally make this smaller than AGENT_MOVEMENT_CONSTANT to improve fidelity
    width=INTEL_CAMERA_WIDTH,
    height=INTEL_CAMERA_HEIGHT,
    visibilityDistance=MAXIMUM_DISTANCE_ARM_FROM_AGENT_CENTER,
    visibilityScheme="Distance",
    fieldOfView=INTEL_VERTICAL_FOV,
    server_class=ai2thor.fifo_server.FifoServer,
    useMassThreshold=True,
    massThreshold=10,
    autoSimulation=False,
    autoSyncTransforms=True,
    renderInstanceSegmentation=True,
    agentMode= AGENT_MODE,
    renderDepthImage=SAVE_DEPTH,
    cameraNearPlane=0.01,  # VERY VERY IMPORTANT
    branch=None,  # IMPORTANT do not use branch
    commit_id=STRETCH_COMMIT_ID,
    server_timeout=MAXIMUM_SERVER_TIMEOUT,
    snapToGrid=False,
    fastActionEmit=True,
    action_hook_runner=_ACTION_HOOK_RUNNER,
    maxDownwardLookAngle=360.0,
    maxUpwardLookAngle=360.0,
)

assert (
    STRETCH_ENV_ARGS.get("branch") is None and STRETCH_ENV_ARGS["commit_id"] is not None
), "Should always specify the commit id and not the branch."


ADDITIONAL_ARM_ARGS = {
    "returnToStart": True,
    "speed": 1,
}

ADDITIONAL_NAVIGATION_ARGS = {
    **ADDITIONAL_ARM_ARGS,
    "returnToStart": False,
}

STRETCH_WRIST_BOUND_1 = 75
STRETCH_WRIST_BOUND_2 = -260
