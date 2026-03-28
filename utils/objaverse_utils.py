import sys
import os

import prior
import compress_json
from objathor.dataset import DatasetSaveConfig, load_holodeck_base

from utils.constants.objaverse_data_dirs import OBJAVERSE_ASSETS_VERSION

OBJAVERSE_HOME_COMMIT_ID = "ace12898b451c887bb1dd69ede85d32a75a86ef7"
# OBJAVERSE_HOME_COMMIT_ID = "877a5d636a6c437b894d1f8510bc852e49bb1cc0" # Has one fix to the above but we'll keep the above for now for backwards compatibility.

if OBJAVERSE_ASSETS_VERSION in ["2025_06_10"]:
    ANNOTATIONS_COMMIT_ID = "39026fb5c1fb8b7b480bd95235d95cb375a47fc2"  # for use with 2025_06_10 assets and holodeck houses
else:
    ANNOTATIONS_COMMIT_ID = OBJAVERSE_HOME_COMMIT_ID

_OBJAVERSE_ANNOTATIONS = {}


def get_objaverse_annotations(revision=ANNOTATIONS_COMMIT_ID):
    global _OBJAVERSE_ANNOTATIONS

    if revision in [
        "ace12898b451c887bb1dd69ede85d32a75a86ef7",
        "877a5d636a6c437b894d1f8510bc852e49bb1cc0",
        "39026fb5c1fb8b7b480bd95235d95cb375a47fc2",
    ]:
        if revision not in _OBJAVERSE_ANNOTATIONS:
            _OBJAVERSE_ANNOTATIONS[revision] = prior.load_dataset(
                "objaverse-plus", revision=revision
            )["train"].data

        return _OBJAVERSE_ANNOTATIONS[revision]
    else:
        raise ValueError(f"Unknown {revision=}")


THOR_ANNOTATIONS_VERSION = "2023_09_23"

THOR_ANNOTATIONS_BASE_PATH = (
    "/data/input/datasets/holodeck_data"
    if sys.platform == "linux"
    else os.path.expanduser(os.path.join("~", ".objathor-assets"))
)

_THOR_ANNOTATIONS = {}


def get_thor_annotations(version=THOR_ANNOTATIONS_VERSION):
    global _THOR_ANNOTATIONS

    if version in ["2023_09_23"]:
        if version not in _THOR_ANNOTATIONS:

            full_path = os.path.join(
                THOR_ANNOTATIONS_BASE_PATH,
                "holodeck",
                version,
                "thor_object_data",
                "annotations.json.gz",
            )

            if not os.path.isfile(full_path):
                dsc = DatasetSaveConfig(
                    VERSION=version,
                    BASE_PATH=THOR_ANNOTATIONS_BASE_PATH,
                )

                path = load_holodeck_base(dsc)

                assert os.path.isfile(
                    full_path
                ), f"thor annotation data installed under {path=} for expected {full_path=}"

            _THOR_ANNOTATIONS[version] = compress_json.load(full_path)

        return _THOR_ANNOTATIONS[version]
    else:
        raise ValueError(f"Unknown {version=}")
