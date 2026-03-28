from typing import Dict, List

from environment.stretch_controller_fpin import FPINController
from environment.stretch_controller_orig import OriginalStretchController
import numpy as np
from utils.constants.stretch_initialization_utils import (
    AGENT_MODE,

)




class StretchController:
    def __new__(cls, **kwargs):
        if AGENT_MODE == 'fpin':
            return FPINController(**kwargs)
        else:
            return OriginalStretchController(**kwargs)
            
    


class StretchStochasticController(StretchController):
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
        return super(StretchStochasticController, self).step(**kwargs)


def filter_good_spawn_locations(spaw_locations: List[Dict[str, float]]) -> List[Dict[str, float]]:
    valid_locations = [
        loc_dict for loc_dict in spaw_locations if loc_dict["y"] > 0.4 and loc_dict["y"] < 0.95
    ]
    return valid_locations