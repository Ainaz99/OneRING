from utils.constants.FPIN_utils import PARAMS_TO_RANDOMIZE


LIST_OF_CAMERAS = {
    'front': 0,
    'left': 1,
    'right': 2,
    'down': 3,
}
INDEX_TO_CAMERA = {v: k for k, v in LIST_OF_CAMERAS.items()}


class AbstractSensor:
    sensor_uuid = "sensor_name"


class ObservationSensor(AbstractSensor):
    pass


class ManipulationCamera(ObservationSensor):
    sensor_uuid = "raw_manipulation_camera"


class NavigationCamera(ObservationSensor):
    sensor_uuid = "raw_navigation_camera"


class NavigationCameraDuplicate(ObservationSensor):
    sensor_uuid = "raw_navigation_camera_2"


class ManipulationCameraDuplicate(ObservationSensor):
    sensor_uuid = "raw_manipulation_camera_2"


VISUAL_SENSORS = [
    ManipulationCamera.sensor_uuid,
    NavigationCamera.sensor_uuid,
    NavigationCameraDuplicate.sensor_uuid,
    ManipulationCameraDuplicate.sensor_uuid,
]

def get_uuids_for_agent_parameter_sensor():
    # order of sensors in the agent parameter sensor is based on these uuids
    return PARAMS_TO_RANDOMIZE

def get_len_agent_parameter_sensor(agent_parameter_uuids):
    sensor_len = get_sensor_sizes()
    return sum([sensor_len[sensor] for sensor in agent_parameter_uuids])

def is_a_visual_sensor(sensor):
    return sensor in VISUAL_SENSORS


def is_a_visual_features(feature):
    og_sensor_name = feature.rsplit("_features", 1)[0]
    return is_a_visual_sensor(og_sensor_name)


def is_a_goal_sensor(sensor):
    goal_sensors = [
        "nav_task_relevant_object_bbox",
        "manip_task_relevant_object_bbox",
        "nav_accurate_object_bbox",
        "manip_accurate_object_bbox",
        "goals",
    ]
    return sensor in goal_sensors


def is_a_goal_features(feature):
    goal_sensors = [
        "nav_task_relevant_object_bbox",
        "manip_task_relevant_object_bbox",
        "nav_accurate_object_bbox",
        "manip_accurate_object_bbox",
        "goal_text_features",
    ]
    return feature in goal_sensors


def is_a_non_visual_sensor(sensor):
    return sensor in [
        "relative_arm_location_metadata",
        "an_object_is_in_hand",
        "last_actions",
        "rooms_seen",
        "room_current_seen",
        "rooms_seen_output",
        "room_current_seen_output",
        "nav_task_relevant_object_bbox",
        "manip_task_relevant_object_bbox",
        "nav_accurate_object_bbox",
        "manip_accurate_object_bbox",
    ]





def get_sensor_sizes():
    return {
        "first_camera_fov": 1,
        "second_camera_fov": 1,
        "first_camera_rotation": 2,
        "second_camera_rotation": 2,
        "width": 1,
        "height": 1,
        "first_camera_position": 3,
        "second_camera_position": 3,
        "agent_body_params": 5, # 2 for offset, 3 for collider scale ratio
    }
