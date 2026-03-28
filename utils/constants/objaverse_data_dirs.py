import os
import warnings



# Dynamic detection of the base data directory based on available paths
def _detect_beaker_objaverse_data_dir():
    """Detect the appropriate Objaverse data directory based on available paths."""
    if os.path.exists("/net/nfs/prior"):
        return "/net/nfs/prior/datasets/vida_datasets/objaverse_assets/2023_07_28"
    elif os.path.exists("/data/input/khzeng"):
        return "/data/input/khzeng/objaverse_assets/2023_07_28_msgpack"
    else:
        # Fallback to the original path
        return "/data/input/khzeng/objaverse_assets/2023_07_28_msgpack"


BEAKER_OBJAVERSE_DATA_DIR = _detect_beaker_objaverse_data_dir()

_OBJAVERSE_DATA_DIR = os.environ.get("OBJAVERSE_DATA_DIR")
OBJAVERSE_ASSETS_DIR = os.environ.get("OBJAVERSE_ASSETS_DIR")
OBJAVERSE_HOUSES_DIR = os.environ.get("OBJAVERSE_HOUSES_DIR")
OBJAVERSE_DATASETS_DIR = os.environ.get("OBJAVERSE_DATASETS_DIR")
OBJAVERSE_ASSETS_VERSION = os.environ.get("OBJAVERSE_DATASETS_DIR")

if any(x is not None for x in [OBJAVERSE_ASSETS_DIR, OBJAVERSE_HOUSES_DIR, OBJAVERSE_DATASETS_DIR]):
    assert OBJAVERSE_ASSETS_DIR is not None
    assert _OBJAVERSE_DATA_DIR is None

    OBJAVERSE_ASSETS_DIR = os.path.abspath(OBJAVERSE_ASSETS_DIR)

    if OBJAVERSE_HOUSES_DIR is None:
        warnings.warn("`OBJAVERSE_HOUSES_DIR` is not set.")
    else:
        OBJAVERSE_HOUSES_DIR = os.path.abspath(OBJAVERSE_HOUSES_DIR)

    if OBJAVERSE_DATASETS_DIR is None:
        pass
        # warnings.warn("`OBJAVERSE_DATASETS_DIR` is not set.")
    else:
        OBJAVERSE_DATASETS_DIR = os.path.abspath(OBJAVERSE_DATASETS_DIR)

    if OBJAVERSE_ASSETS_VERSION is None:
        OBJAVERSE_ASSETS_VERSION = "2023_07_28"
        warnings.warn(f"Enforced missing {OBJAVERSE_ASSETS_VERSION=} with {OBJAVERSE_ASSETS_DIR=}")
else:
    if _OBJAVERSE_DATA_DIR is None:
        if os.path.exists(BEAKER_OBJAVERSE_DATA_DIR):
            print(f"Using {BEAKER_OBJAVERSE_DATA_DIR} as base dir for objaverse houses/assets.")
            _OBJAVERSE_DATA_DIR = BEAKER_OBJAVERSE_DATA_DIR
        else:
            print(
                f"Missing `OBJAVERSE_DATA_DIR` environment variable. Setting OBJAVERSE_DATA_DIR={os.getcwd()}"
            )
            _OBJAVERSE_DATA_DIR = os.getcwd()

    _OBJAVERSE_DATA_DIR = os.path.abspath(_OBJAVERSE_DATA_DIR)

    # version: 2023_07_28
    OBJAVERSE_HOUSES_DIR = os.path.join(_OBJAVERSE_DATA_DIR, "houses_2023_07_28/")
    OBJAVERSE_ASSETS_DIR = os.path.join(_OBJAVERSE_DATA_DIR, "processed_2023_07_28/")
    OBJAVERSE_DATASETS_DIR = os.path.join(_OBJAVERSE_DATA_DIR, "procthor_databases_2023_07_28/")
    OBJAVERSE_BODY_ASSETS_DIR = os.path.join(_OBJAVERSE_DATA_DIR, "2024_03_01", "assets")
    OBJAVERSE_ASSETS_VERSION = "2023_07_28"

    # version: 2025_06_10
    # OBJAVERSE_HOUSES_DIR = os.path.join(_OBJAVERSE_DATA_DIR, "houses/")
    # OBJAVERSE_ASSETS_DIR = os.path.join(_OBJAVERSE_DATA_DIR, "assets/")
    # OBJAVERSE_DATASETS_DIR = None
    # OBJAVERSE_BODY_ASSETS_DIR = os.path.join(_OBJAVERSE_DATA_DIR, "2024_03_01", "assets")
    # OBJAVERSE_ASSETS_VERSION = "2025_06_10"

print(
    f"Using {OBJAVERSE_ASSETS_DIR=}\n"
    f"      {OBJAVERSE_HOUSES_DIR=}\n"
    f"      {OBJAVERSE_DATASETS_DIR=}\n"
    f"      {OBJAVERSE_ASSETS_VERSION=}"
)
