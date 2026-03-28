import os
from pathlib import Path

WANDB_DEFAULT_PROJECT_NAME = "fpin"
WANDB_DEFAULT_DATA_PROJ = "spoc-v2-data"
ABS_PATH_OF_VIDA_DIR = os.path.abspath(os.path.dirname(Path(__file__)))

DATA_MODE = "train"

GO_NEAR_POINT_SUCCESS_DIST_THRESHOLD = 1
GO_NEAR_POINT_SAMPLING_MIN_PIXEL_VISIBLE_THRESHOLD = 50
GO_NEAR_POINT_SAMPLING_MIN_WH_THRESHOLD = 5
MIN_HEIGHT_OF_OBJECT_TO_SAMPLE_FOR_PICKUP = 0.2
MIN_NUM_OPEN_SPOTS_ON_RECEPTACLE = 5

if DATA_MODE == "train":
    GO_NEAR_POINT_SAMPLING_DIST_THRESHOLD = 20
    PICKUP_POINT_SAMPLING_DIST_THRESHOLD = 5
elif DATA_MODE == "benchmark":
    GO_NEAR_POINT_SAMPLING_DIST_THRESHOLD = 10
    PICKUP_POINT_SAMPLING_DIST_THRESHOLD = 1.5
else:
    raise ValueError(f"DATA_MODE {DATA_MODE} not recognized")


RECEPTACLE_LIST_FOR_DROP = [
    "Bed",
    # "Chair",
    "Counter",
    "CounterTop",
    "Desk",
    "DiningTable",
    "Dresser",
    "Drawer",
    "Shelf",
    "Sink",
    "SinkBasin",
    "Sofa",
    "Table",
    "Toilet",
    "TVStand",
]
RECEPTACLE_LIST_FOR_DROP = RECEPTACLE_LIST_FOR_DROP + [f"Obja{i}" for i in RECEPTACLE_LIST_FOR_DROP]

LIST_OF_PICKUPABLE_OBJECTS = [
    "AlarmClock",
    # "AluminumFoil",
    "Apple",
    "AppleSliced",
    "Book",
    "Boots",
    "Bottle",
    "Bowl",
    "Bread",
    # "BreadSliced",
    "Candle",
    # "CellPhone",
    # "Cloth",
    # "CreditCard",
    "Cup",
    "DishSponge",
    "Dumbbell",
    "Egg",
    # "EggCracked",
    # "Fork",
    "HandTowel",
    "Kettle",
    # "KeyChain",
    "Knife",
    "Ladle",
    "Laptop",
    "Lettuce",
    # "LettuceSliced",
    "Mug",
    "Newspaper",
    "Pan",
    "PaperTowelRoll",
    # "Pen",
    # "Pencil",
    "PepperShaker",
    "Pillow",
    "Plate",
    # "Plunger",
    "Pot",
    "Potato",
    # "PotatoSliced",
    "RemoteControl",
    "SaltShaker",
    "ScrubBrush",
    # "SoapBar",
    "SoapBottle",
    # "Spatula",
    # "Spoon",
    "SprayBottle",
    "Statue",
    "TableTopDecor",
    "TeddyBear",
    "TissueBox",
    "ToiletPaper",
    "Tomato",
    # "TomatoSliced",
    "Towel",
    "Vase",
    # "Watch",
    "WineBottle",
]
LIST_OF_PICKUPABLE_OBJECTS = LIST_OF_PICKUPABLE_OBJECTS + [
    f"Obja{i}" for i in LIST_OF_PICKUPABLE_OBJECTS
]

MIN_DISTANCE_TO_RECEPTACLE_FOR_DROP = 2
MAX_DISTANCE_TO_RECEPTACLE_FOR_DROP = 4 
VISIBILITY_DIST_FOR_RECEPTACLE = 5
