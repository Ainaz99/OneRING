from inspect import isclass
from typing import Dict

import data_generation.path_planners as path_planners
from data_generation.path_planners import PathPlanner
from utils.type_utils import REGISTERED_TASK_PARAMS

REGISTERED_PLANNERS: Dict[str, type] = {}


def register_planner(cls):
    # ignore planners without task_type_str
    if cls.task_type_str is None:
        return cls

    for task_str in cls.task_type_str:
        # ignore task_type_str not registered in REGISTERED_TASK_PARAMS
        if task_str not in REGISTERED_TASK_PARAMS:
            continue
        REGISTERED_PLANNERS[task_str] = cls
    return cls


# import the module and iterate through its attributes
for attribute_name in dir(path_planners):
    attribute = getattr(path_planners, attribute_name)
    if isclass(attribute):
        # Add the class to this package's variables
        if issubclass(attribute, PathPlanner) and attribute != PathPlanner:
            globals()[attribute_name] = attribute
            register_planner(attribute)
