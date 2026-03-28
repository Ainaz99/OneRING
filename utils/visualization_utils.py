import copy
import traceback
from typing import Sequence, Dict

import numpy as np

from environment.stretch_controller import StretchController

DISTINCT_COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (128, 0, 0),  # Dark Red
    (0, 128, 0),  # Dark Green
    (0, 0, 128),  # Dark Blue
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (255, 165, 0),  # Orange
    (255, 192, 203),  # Pink
    (255, 255, 255),  # White
    (0, 0, 0),  # Black
    (0, 0, 139),  # DarkBlue
    (0, 100, 0),  # DarkGreen
    (139, 0, 139),  # DarkMagenta
    (165, 42, 42),  # Brown
    (255, 215, 0),  # Gold
    (64, 224, 208),  # Turquoise
    (240, 230, 140),  # Khaki
    (70, 130, 180),  # Steel Blue
]



def get_top_down_path_view(
    controller: StretchController,
    agent_path: Sequence[Dict[str, float]],
    targets_to_highlight=None,
    orthographic: bool = True,
    map_height_width=(1000, 1000),
    path_width: float = 0.045,
):
    thor_controller = controller.controller

    original_hw = thor_controller.last_event.frame.shape[:2]

    if original_hw != map_height_width:
        event = thor_controller.step(
            "ChangeResolution", x=map_height_width[1], y=map_height_width[0], raise_for_failure=True
        )

    if len(thor_controller.last_event.third_party_camera_frames) < 2:
        event = thor_controller.step("GetMapViewCameraProperties", raise_for_failure=True)
        cam = copy.deepcopy(event.metadata["actionReturn"])
        if not orthographic:
            bounds = event.metadata["sceneBounds"]["size"]
            max_bound = max(bounds["x"], bounds["z"])

            cam["fieldOfView"] = 50
            cam["position"]["y"] += 1.1 * max_bound
            cam["orthographic"] = False
            cam["farClippingPlane"] = 50
            del cam["orthographicSize"]

        event = thor_controller.step(
            action="AddThirdPartyCamera",
            **cam,
            skyboxColor="white",
            raise_for_failure=True,
        )

    waypoints = []
    for target in targets_to_highlight or []:
        target_position = controller.get_object_position(target)
        target_dict = {
            "position": target_position,
            "color": {"r": 1, "g": 0, "b": 0, "a": 1},
            "radius": 0.5,
            "text": "",
        }
        waypoints.append(target_dict)

    if len(agent_path) != 0:
        thor_controller.step(
            action="VisualizeWaypoints",
            waypoints=waypoints,
            raise_for_failure=True,
        )
        # put this over the waypoints just in case
        event = thor_controller.step(
            action="VisualizePath",
            positions=agent_path,
            pathWidth=path_width,
            raise_for_failure=True,
        )
        thor_controller.step({"action": "HideVisualizedPath"})

    map = event.third_party_camera_frames[-1]

    if original_hw != map_height_width:
        thor_controller.step(
            "ChangeResolution", x=original_hw[1], y=original_hw[0], raise_for_failure=True
        )

    return map


from PIL import Image, ImageDraw, ImageFont


def create_multiline_text_image(
    text,
    width,
    height,
    bg_color=(255, 255, 255),
    text_color=(0, 0, 0),
    font_path=None,
    font_size=20,
):
    # Create a blank image with the background color
    image = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Load a font
    font = ImageFont.truetype(font_path if font_path else "Arial.ttf", font_size)

    # Split the text into lines
    lines = text.split("\n")
    line_heights = [30 for line in lines]
    total_text_height = sum(line_heights) + (len(lines) - 1) * 10  # add spacing between lines

    # Calculate initial y position (top margin)
    y = (height - total_text_height) / 2

    for line in lines:
        # Calculate text width and height using textbbox for precise bounding box
        text_width, text_height = draw.textbbox((0, 0), line, font=font)[2:]

        # Calculate x position (left margin)
        x = (width - text_width) / 2

        # Draw each line
        draw.text((x, y), line, font=font, fill=text_color)
        y += text_height + 10  # move y to the next line position with spacing

    # Convert the PIL Image to a numpy array
    numpy_image = np.array(image)

    return numpy_image


