import json
from typing import TypedDict, Literal, List, Dict, Any, Union, TYPE_CHECKING

import numpy as np

from utils.type_utils import REGISTERED_TASK_PARAMS

if TYPE_CHECKING:
    from tasks.task_specs import TaskSpec

import hashlib


def generate_deterministic_task_id(task_data: Dict[str, Any]) -> str:
    """
    Generate a deterministic task ID based purely on the task content.

    Args:
        task_data: Dictionary containing task information

    Returns:
        Deterministic task ID string
    """
    # Extract core identifiers
    task_type = task_data.get("task_type", "unknown")
    house_index = task_data.get("house_index", 0)

    # Get starting position - try common formats
    pos = task_data.get("agent_starting_position", [0, 0, 0])
    if isinstance(pos, dict):
        pos = [pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)]
    elif "observations" in task_data and "initial_agent_location" in task_data["observations"]:
        initial_loc = task_data["observations"]["initial_agent_location"]
        if hasattr(initial_loc, "tolist"):
            initial_loc = initial_loc.tolist()
        if isinstance(initial_loc, (list, tuple)) and len(initial_loc) >= 3:
            pos = initial_loc[:3]

    # Get rotation
    rotation = task_data.get("agent_y_rotation", 0)
    if rotation is None and "observations" in task_data:
        initial_loc = task_data["observations"].get("initial_agent_location", [])
        if isinstance(initial_loc, (list, tuple)) and len(initial_loc) >= 5:
            rotation = initial_loc[4]


    # Create deterministic content dictionary
    content = {
        "task_type": task_type,
        "house": house_index,
        "pos": [round(float(p), 2) for p in pos[:3]] if len(pos) >= 3 else [0, 0, 0],
        "rot": round(float(rotation), 2) if rotation is not None else 0,
    }

    # Generate hash from sorted JSON representation
    content_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
    content_hash = hashlib.md5(content_str.encode("utf-8")).hexdigest()[:12]

    # Create readable task ID
    return f"{task_type}_h{house_index}_{content_hash}"






class EvalSample(TypedDict):
    task_type: str
    house_index: int
    natural_language_spec: str

    agent_starting_position: List[float]
    agent_y_rotation: float

    expert_length_bucket: Literal["long", "medium", "short"]
    expert_length: int
    synsets: List[str]
    synset_to_object_ids: Dict[str, List[str]]
    broad_synset_to_object_ids: Dict[str, List[str]]
    extras: Dict[str, Any]
    task_path: str
    hypernyms: List[str]


class Observations(TypedDict):
    goal: str
    initial_agent_location: Union[np.ndarray, List[float]]  # 6 floats (xyz + 0rotation0)
    actions: List[str]
    time_ids: List[int]
    templated_task_type: str


class NormalizedEvalSample(TypedDict):
    task_type: str
    house_id: str

    sample_id: str

    sub_house_id: int
    needs_video: bool
    raw_navigation_camera: str
    sensors_path: str

    observations: Observations


def eval_sample_to_normalized_eval_sample(
    task_type: str, sample: EvalSample, index: int
) -> NormalizedEvalSample:
    if "task_type" in sample:
        assert task_type == sample["task_type"]

    # Create a task_data dict compatible with generate_deterministic_task_id
    task_data = {
        "task_type": task_type,
        "house_index": sample["house_index"],
        "agent_starting_position": sample["agent_starting_position"],
        "agent_y_rotation": sample["agent_y_rotation"],
    }

    # Generate deterministic sample ID using the unified function
    sample_id = generate_deterministic_task_id(task_data)

    return NormalizedEvalSample(
        sample_id=sample_id,
        house_id=str(sample["house_index"]).zfill(6),
        task_type=task_type,
        sub_house_id=index,
        needs_video=False,
        raw_navigation_camera="",
        sensors_path="",
        observations=Observations(
            goal=sample["natural_language_spec"],
            initial_agent_location=np.array(
                sample["agent_starting_position"] + [0, sample["agent_y_rotation"], 0]
            ),
            actions=[],
            time_ids=[],
            templated_task_type=json.dumps(sample),
        ),
    )


def normalized_eval_sample_to_task_spec(s: NormalizedEvalSample) -> "TaskSpec":
    templated_task_info = json.loads(s["observations"]["templated_task_type"])
    task_spec = {
        "task_type": s["task_type"],
        "house_index": int(s["house_id"]),
        "natural_language_spec": s["observations"]["goal"],
        "agent_starting_position": s["observations"]["initial_agent_location"][:3],
        "agent_y_rotation": float(s["observations"]["initial_agent_location"][-2]),
        "eval_info": {
            "sample_id": s["sample_id"],
            "needs_video": s["needs_video"],
            **templated_task_info,
        },
    }

    for key in REGISTERED_TASK_PARAMS.get(s["task_type"]):
        try:
            task_spec[key] = templated_task_info[key]
        except KeyError:
            raise KeyError(
                f"Key {key} not found in {templated_task_info}, but is required by {s['task_type']}"
            )

    return task_spec
