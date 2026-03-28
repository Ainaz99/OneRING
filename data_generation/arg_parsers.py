import argparse

import torch.cuda

from allenact.utils.misc_utils import str2bool
from tasks import REGISTERED_TASKS


def create_basic_argument_parser(description="Generating an offline dataset"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--save_dir",
        type=str,
        help="""The directory to save the dataset to.""",
        required=True,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="""Name of the dataset.""",
        required=True,
    )
    parser.add_argument(
        "--split", type=str, help="""The train, val, or test split.""", required=True
    )
    parser.add_argument(
        "--stochastic_controller",
        type=bool,
        help="""Whether we use the stochastic variant of the controller.""",
        required=False,
        default=True,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="""The maximum number of steps in dataset generation.""",
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--closed_type",
        type=str2bool,
        default=False,
        help="""Limit target object types for backwards comparisons.""",
        required=False,
    )
    parser.add_argument(
        "--house_dataset",
        help="""Which house dataset to use, 'procthor' or 'objaverse'.""",
        default="objaverse",
    )
    parser.add_argument(
        "--debug",
        type=str2bool,
        help="""Whether or not to use the debug house set of the above house dataset.""",
        default=not torch.cuda.is_available(),
    )
    return parser


def add_action_space_args(parser):
    parser.add_argument(
        "--action_space_type",
        type=str,
        help="""The action space type to use (spoc, continuous, quantized).""",
        required=True,
    )
    parser.add_argument(
        "--action_space_name",
        type=str,
        help="""The action space subtype name to use (default, spocbounds, etc. See .ini files)""",
        required=False,
        default="default",
    )
    return parser


# def parse_args_for_sqs_manager():
#     parser = create_basic_argument_parser()
#     parser.add_argument(
#         "--sqs_queue_name", type=str, help="""The SQS queue to use.""", required=True
#     )
#     parser.add_argument(
#         "--metrics_json_dir",
#         type=str,
#         help="""Place to save beaker metrics.json file.""",
#         required=False,
#         default=None,
#     )
#     return parser.parse_args()


# def parse_args_for_mpqueue_manager():
#     parser = create_basic_argument_parser()
#     parser.add_argument(
#         "--house_repeats",
#         type=int,
#         help="""Number of trajectories to generate per house.""",
#         required=True,
#     )
#     return parser.parse_args()


def parse_args_for_multi_task_mpqueue_manager():
    parser = create_basic_argument_parser()
    parser.add_argument(
        "--task_type",
        type=str,
        help=f"""Registered type of task (see tasks.REGISTERED_TASKS=={REGISTERED_TASKS}).""",
        required=True,
    )
    parser.add_argument(
        "--house_repeats",
        type=int,
        help="""Number of trajectories to generate per house.""",
        required=True,
    )
    parser = add_action_space_args(parser)
    return parser.parse_args()


def parse_args_for_multi_task_sqs_manager():
    parser = create_basic_argument_parser()
    parser.add_argument(
        "--task_type",
        type=str,
        help=f"""Registered type of task (see tasks.REGISTERED_TASKS=={REGISTERED_TASKS}).""",
        required=True,
    )
    parser.add_argument(
        "--sqs_queue_name", type=str, help="""The SQS queue to use.""", required=True
    )
    parser.add_argument(
        "--metrics_json_dir",
        type=str,
        help="""Place to save beaker metrics.json file.""",
        required=False,
        default=None,
    )
    parser = add_action_space_args(parser)
    return parser.parse_args()
