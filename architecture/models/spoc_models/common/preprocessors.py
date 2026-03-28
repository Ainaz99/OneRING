import json
import random
import traceback
import gym
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union, Any, Dict, cast, Sequence

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image
from open_clip import get_tokenizer
from torch.distributions.utils import lazy_property
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Compose, Normalize
from transformers import AutoTokenizer

from utils.constants.FPIN_utils import DEFAULT_ASSET
from utils.sensor_constant_utils import is_a_visual_sensor
from utils.transformation_util import (
    get_full_transformation_list,
    sample_a_specific_transform,
)
from utils.string_utils import convert_byte_to_string
from environment.action_spaces import AbstractActionSpace
from allenact.base_abstractions.preprocessor import Preprocessor as AllenActPreprocessor
from architecture.models.spoc_models.common.image_encoder import IMAGE_ENCODER
from architecture.models.spoc_models.common.text_encoder import TEXT_ENCODER


def mask_agent_in_frame(frames, mask):
    T = frames.shape[0]
    random_color = torch.randint(0, 256, (T, 1, 1, 3), dtype=frames.dtype, device=frames.device)
    mask = mask.bool()
    mask = mask.expand_as(frames).detach()
    frames = torch.where(mask, random_color, frames)
    # if mask.sum().item() > 0:
    #     for i in range(T):
    #         array = frames[i].cpu().numpy()
    #         image = Image.fromarray(array)
    #         image.save(f'output_image_{i}.png')
    return frames



def tensor_image_preprocessor(
    size=(224, 384),
    data_augmentation=False,
    specific=False,
    augmentation_version="v2",
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
    padding=True
):
    def convert_to_float(tensor):
        return tensor.float() / 255.0

    list_of_transformations = []


    # resize without padding
    list_of_transformations += [
        torchvision.transforms.Resize(
            size,
            interpolation=T.InterpolationMode("bicubic"),
            max_size=None,
            antialias=True,
        )
    ]

    if data_augmentation:
        data_aug_transforms = get_full_transformation_list(size=size, version=augmentation_version)
        if specific:
            data_aug_transforms = sample_a_specific_transform(
                Compose(data_aug_transforms)
            ).transforms

        list_of_transformations += data_aug_transforms

    list_of_transformations += [
        torchvision.transforms.Lambda(convert_to_float),
        Normalize(mean=mean, std=std),
    ]
    return Compose(list_of_transformations)


@dataclass
class PreprocessorConfig:
    max_steps: int = None
    pad: bool = True
    action_space: AbstractActionSpace = None
    data_augmentation: bool = True
    augmentation_version: str = "v2"
    model_version: str = ""
    text_encoder_context_length: int = None
    image_encoder: IMAGE_ENCODER = IMAGE_ENCODER.Dinov2Small
    text_encoder: TEXT_ENCODER = TEXT_ENCODER.T5Small
    goal_text_padding_length: int = 0

    @property
    def mean(self):
        return self.image_encoder.value[1].mean

    @property
    def stdev(self):
        return self.image_encoder.value[1].stdev

    @property
    def image_size(self) -> Tuple[int, int]:
        return self.image_encoder.value[1].input_size




class Preprocessor:
    def __init__(self, cfg: PreprocessorConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.img_enc_cls, self.img_enc_cfg = cfg.image_encoder.value
        self._image_encoder = None
        self.text_enc_cls, self.text_enc_cfg = cfg.text_encoder.value
        self._text_encoder = None
        self.to(self.device)
        # self.i = 0

    @property
    def image_encoder(self):
        if self._image_encoder is None:
            self._image_encoder = self.img_enc_cls(self.img_enc_cfg)
            self._image_encoder.eval()
        return self._image_encoder

    @property
    def text_encoder(self):
        if self._text_encoder is None:
            self._text_encoder = self.text_enc_cls(self.text_enc_cfg)
            self._text_encoder.eval()
        return self._text_encoder

    def to(self, device: torch.device):
        self.device = device
        self._image_encoder = self.image_encoder.to(device)
        self._text_encoder = self.text_encoder.to(device)

    @lazy_property
    def image_preprocessor(self):
        return tensor_image_preprocessor(
            size=self.cfg.image_size,
            data_augmentation=self.cfg.data_augmentation,
            augmentation_version=self.cfg.augmentation_version,
            mean=self.cfg.mean,
            std=self.cfg.stdev,
            padding=True
        )

    @lazy_property
    def text_preprocessor(self):
        return AutoTokenizer.from_pretrained("t5-small")

    def process_frames(self, batch, sensor_key):

        frame_processor = self.get_frame_processor(sensor_key)
        frames = list(map(frame_processor, batch))

        if self.cfg.pad:
            return pad_sequence(frames, batch_first=True, padding_value=0)

        return frames

    def get_frame_processor(self, sensor_key):

        def frame_processor(sample):
            frames = sample[sensor_key]



            frames = frames[: self.cfg.max_steps].to(self.device)
            frames = frames.permute(0, 3, 1, 2)
            # TODO remove this after the data generation is fixed
            try:
                res = self.image_preprocessor(frames)

            except Exception as e:
                print("Exception in frame preprocessor")
                print(e)
                print(traceback.format_exc())
                print("sensor_key", sensor_key)
                print("self.cfg.max_steps", self.cfg.max_steps)
                print("after permute frames.shape", frames.shape)
                print("before permute", sample[sensor_key][: self.cfg.max_steps].shape)
            return res

        return frame_processor

    def compute_image_feature(self, frames):
        # Move input to the image encoder device
        frames = frames.to(self.device)
        # frames are in B, T, C, H, W dim
        b, t, c, h, w = frames.shape
        
        frames = torch.reshape(frames, (-1, *frames.shape[2:]))
        # frames are now in BT, C, H, W dim
        features = self.image_encoder(frames)
        # features are now in BT, D, H', W' dim
        features = torch.reshape(features, (b, t, *features.shape[1:]))
        # features are now in B, T, D, H', W' dim
        return features

    @property
    def num_actions(self):
        return self.cfg.action_space.get_num_actions()  # 20

    def process_actions(self, batch):
        action_processor = self.get_action_processor()
        actions = list(map(action_processor, batch))
        if self.cfg.pad:
            return pad_sequence(actions, batch_first=True, padding_value=-1)
        return actions

    def get_action_processor(self):
        action_processor = self.get_action_processor_generic("actions")
        return action_processor

    def process_last_actions(self, batch):
        last_actions_processor = self.get_last_actions_processor()
        last_actions = list(map(last_actions_processor, batch))
        if self.cfg.pad:
            return pad_sequence(
                last_actions, batch_first=True, padding_value=self.num_actions
            )  # 0-19 are actions, 20 is for padding, 21 is for start token ("")

        return last_actions

    def get_action_processor_generic(self, key):
        def get_action_processor(sample):
            last_actions = sample[key][: self.cfg.max_steps]
            action_space = self.cfg.action_space
            if action_space.is_tokenized:
                last_actions = [action_space.tokenizer.get_token_id(x) for x in last_actions]
            else:
                last_actions = [action_space.get_action_index_from_string(x) for x in last_actions]
            last_actions = torch.tensor(last_actions, dtype=torch.int64).to(self.device)
            return last_actions

        return get_action_processor

    def get_last_actions_processor(self):
        last_actions_processor = self.get_action_processor_generic("last_actions")
        return last_actions_processor

    def process_goals(self, batch):
        goal_spec = self.text_preprocessor(
            [sample["goal"] for sample in batch],
            return_tensors="pt",
            padding=True,
        )
        if self.cfg.goal_text_padding_length > goal_spec.data["input_ids"].shape[1]:
            goal_spec.data["input_ids"] = torch.nn.functional.pad(
                goal_spec.data["input_ids"],
                (0, self.cfg.goal_text_padding_length - goal_spec.data["input_ids"].shape[1]),
            )
            goal_spec.data["attention_mask"] = torch.nn.functional.pad(
                goal_spec.data["attention_mask"],
                (
                    0,
                    self.cfg.goal_text_padding_length - goal_spec.data["attention_mask"].shape[1],
                ),
                mode="constant",
                value=1,
            )
        return {k: v.to(self.device) for k, v in goal_spec.items()}

    def compute_text_feature(self, goals):
        # Move input to the text encoder device
        if isinstance(goals, torch.Tensor):
            goals = goals.to(self.device)
            return self.text_encoder(goals)
        else:
            goals = {k: v.to(self.device) for k, v in goals.items()}
            return self.text_encoder(goals)

    def process_visibility(self, batch):
        visibility = [torch.tensor(sample["visibility"]) for sample in batch]
        if self.cfg.pad:
            return pad_sequence(visibility, batch_first=True, padding_value=-1).to(self.device)

        return visibility

    def process_rooms_seen(self, batch, key="rooms_seen"):
        rooms_seen = [torch.tensor(sample[key]) for sample in batch]
        if self.cfg.pad:
            return pad_sequence(rooms_seen, batch_first=True, padding_value=19).to(self.device)

        return rooms_seen

    def process_room_current_seen(self, batch, key="room_current_seen"):
        room_current_seen = [torch.tensor(sample[key], dtype=torch.int64) for sample in batch]
        if self.cfg.pad:
            return pad_sequence(room_current_seen, batch_first=True, padding_value=2).to(
                self.device
            )

        return room_current_seen

    def process_time_ids(self, batch):
        time_ids = [torch.tensor(sample["time_ids"][: self.cfg.max_steps]) for sample in batch]
        if self.cfg.pad:
            return pad_sequence(time_ids, batch_first=True, padding_value=-1).to(self.device)

        return time_ids

    def process_objinhand(self, batch):
        obj_in_hand = [
            torch.tensor(sample["an_object_is_in_hand"][: self.cfg.max_steps]).long()
            for sample in batch
        ]
        if self.cfg.pad:
            return pad_sequence(obj_in_hand, batch_first=True, padding_value=2).to(self.device)

        return obj_in_hand

    def process_traj_index(self, batch):
        time_ids = [torch.tensor(sample["traj_index"][: self.cfg.max_steps]) for sample in batch]
        if self.cfg.pad:
            return pad_sequence(time_ids, batch_first=True, padding_value=-1).to(self.device)

        return time_ids

    def process_arm_proprioceptive(self, batch):
        arm_proprioceptive = [
            torch.tensor(sample["relative_arm_location_metadata"][: self.cfg.max_steps]).float()
            for sample in batch
        ]
        if self.cfg.pad:
            return pad_sequence(arm_proprioceptive, batch_first=True, padding_value=-1).to(
                self.device
            )
        else:
            return torch.Tensor(arm_proprioceptive).to(self.device)

    def process_last_action_success(self, batch):
        last_action_success = [torch.tensor(sample["last_action_success"]) for sample in batch]
        if self.cfg.pad:
            return pad_sequence(last_action_success, batch_first=True, padding_value=-1).to(
                self.device
            )
        else:
            return torch.Tensor(last_action_success).to(self.device)
        
    def process_agent_parameter(self, batch):
        first_element = batch[0]["agent_parameter_sensor"]
        if isinstance(first_element, torch.Tensor):
            agent_parameter = (
                torch.cat([torch.tensor(sample["agent_parameter_sensor"]) for sample in batch])
                .float()
                .to(self.device)
            )
        else:
            agent_parameter = (
                torch.tensor(np.array([sample["agent_parameter_sensor"] for sample in batch]))
                .float()
                .to(self.device)
            )
        return agent_parameter

        

    def create_padding_mask(self, lengths, max_length):
        # Create a range tensor with the shape (1,max_length)
        range_tensor = torch.arange(max_length, device=self.device).unsqueeze(0)
        return range_tensor >= lengths.unsqueeze(1)

    def process_goal_agent_frame(self, batch):
        goal_agent_frame = [torch.tensor((sample["goal_agent_frame"])) for sample in batch]
        if self.cfg.pad:
            return pad_sequence(goal_agent_frame, batch_first=True, padding_value=-1).to(
                self.device
            )
        return goal_agent_frame

    def process(self, batch, compute_image_feature_once=False):
        if len(batch) == 0:
            return None

        batch = [sample["input_sensors"] for sample in batch]

        batch_keys = list(batch[0].keys())
        output = dict()


        processed_visual_sensors = []
        processed_visual_sensors_keys = []
        for sensor in batch_keys:
            if is_a_visual_sensor(sensor):
                output[sensor] = self.process_frames(batch, sensor_key=sensor)
                if compute_image_feature_once:
                    processed_visual_sensors.append(output[sensor])
                    processed_visual_sensors_keys.append(sensor)
                else:
                    output[f"{sensor}_features"] = self.compute_image_feature(output[sensor])
            elif sensor == "an_object_is_in_hand":
                output[sensor] = self.process_objinhand(batch)
            elif sensor == "agent_parameter_sensor":
                output[sensor] = self.process_agent_parameter(batch)
            elif sensor == "relative_arm_location_metadata":
                output[sensor] = self.process_arm_proprioceptive(batch)
            elif sensor == "actions":
                output["actions"] = self.process_actions(batch)
            elif sensor == "goal_agent_frame":
                output["goal_agent_frame"] = self.process_goal_agent_frame(batch)
            elif sensor == "last_actions":
                output["last_actions"] = self.process_last_actions(batch)
            elif sensor == "last_action_success":
                output["last_action_success"] = self.process_last_action_success(batch)
            elif sensor == "goal":
                output["goal_text_tokens"] = self.process_goals(batch)
                output["goal_text_features"] = self.compute_text_feature(output["goal_text_tokens"])
            elif sensor == "time_ids":
                output["time_ids"] = self.process_time_ids(batch)
            elif sensor == "traj_index":
                output["traj_index"] = self.process_traj_index(batch)
            elif sensor == "visibility":
                output["visibility"] = self.process_visibility(batch)
            elif sensor in ["rooms_seen", "rooms_seen_output"]:
                output[sensor] = self.process_rooms_seen(batch, key=sensor)
            elif sensor in ["room_current_seen", "room_current_seen_output"]:
                output[sensor] = self.process_room_current_seen(batch, key=sensor)
            else:
                if sensor not in ["initial_agent_location", "templated_task_type"]:
                    raise NotImplementedError(f"Sensor {sensor} not implemented")

        if compute_image_feature_once:
            feats = self.compute_image_feature(torch.cat(processed_visual_sensors, dim=0))
            b = feats.shape[0] // len(processed_visual_sensors)
            for i, sensor in enumerate(processed_visual_sensors_keys):
                output[f"{sensor}_features"] = feats[i * b : (i + 1) * b]

        if "actions" in batch_keys:
            key_to_look_at = "actions"
        elif "last_actions" in batch_keys:
            key_to_look_at = "last_actions"
        else:
            key_to_look_at = random.choice([k for k in batch_keys if is_a_visual_sensor(k)])

        output["lengths"] = torch.tensor(
            [len(sample[key_to_look_at]) for sample in batch], dtype=torch.int32
        ).to(self.device)

        if self.cfg.pad:
            output["padding_mask"] = self.create_padding_mask(
                output["lengths"], output[key_to_look_at].shape[1]
            )

        return output


@dataclass
class SigLipPreprocessorConfig(PreprocessorConfig):
    model_version: str = "hf-hub:timm/ViT-B-16-SigLIP-256"
    text_encoder_context_length: int = 64

    image_encoder: IMAGE_ENCODER = IMAGE_ENCODER.SigLIPBase
    text_encoder: TEXT_ENCODER = TEXT_ENCODER.SigLIPBase

@dataclass
class SigLipPreprocessorNoImagePaddingConfig(SigLipPreprocessorConfig):
    image_padding: bool = False


class SigLipPreprocessor(Preprocessor):
    @lazy_property
    def image_preprocessor(self):
        return tensor_image_preprocessor(
            size=self.cfg.image_size,
            data_augmentation=self.cfg.data_augmentation,
            augmentation_version=self.cfg.augmentation_version,
            mean=self.cfg.mean,
            std=self.cfg.stdev,
            padding=True
        )

    @lazy_property
    def text_preprocessor(self):
        return get_tokenizer(self.cfg.model_version)

    def process_goals(self, batch):
        goal_spec = self.text_preprocessor(
            [sample["goal"] for sample in batch],
            context_length=self.cfg.text_encoder_context_length,  # for SigLIP
        )
        return goal_spec.to(self.device)


class SigLipWContinuousActionPreprocessor(SigLipPreprocessor):
    @property
    def num_actions(self):
        raise NotImplementedError("This method should be implemented in the subclass")
        # return self.cfg.action_space.get_num_actions()

    def process_actions(self, batch):
        action_processor = self.get_action_processor()
        actions = list(map(action_processor, batch))
        if self.cfg.pad:
            return pad_sequence(actions, batch_first=True, padding_value=-1)
        return actions

    def get_action_processor(self):
        action_processor = self.get_action_processor_generic("actions")
        return action_processor

    def process_last_actions(self, batch):
        last_actions_processor = self.get_last_actions_processor()
        last_actions = list(map(last_actions_processor, batch))
        if self.cfg.pad:
            return pad_sequence(
                last_actions,
                batch_first=True,
                padding_value=self.cfg.action_space.get_num_actions(),
            )  # KE: This value is the max number of actions because we need to embed that

        return last_actions

    def get_action_processor_generic(self, key):
        def get_action_processor(sample):
            last_actions = sample[key][: self.cfg.max_steps]
            action_space = self.cfg.action_space

            last_actions = [
                action_space.get_simple_action_vector_from_string(x) for x in last_actions
            ]
            last_actions = torch.tensor(last_actions, dtype=torch.int64).to(self.device)
            return last_actions

        return get_action_processor

    def get_last_actions_processor(self):
        last_actions_processor = self.get_action_processor_generic("last_actions")
        return last_actions_processor


@dataclass
class RLUnifiedPreprocessorConfig(SigLipPreprocessorConfig):
    rgb_input_uuid: List[str] = field(default_factory=lambda: [])
    text_input_uuid: str = ""
    goal_text_uuid: str = ""
    output_uuid: str = ""
    feats_uuid: List[str] = field(default_factory=lambda: [])
    goal_text_padding_length: int = 0

    @property
    def output_shape(self):
        return self.image_encoder.value[1].output_size


class RLUnifiedPreprocessor(AllenActPreprocessor):
    def __init__(
        self,
        cfg: RLUnifiedPreprocessorConfig,
        preprocessor: Preprocessor,
    ):
        self.cfg = cfg
        self.preprocessor = preprocessor
        super().__init__(
            input_uuids=cfg.rgb_input_uuid + [cfg.text_input_uuid],
            output_uuid=cfg.output_uuid,
            observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=cfg.output_shape),
        )

    def to(self, device: torch.device) -> "RLUnifiedPreprocessor":
        self.preprocessor.to(device)
        return self

    def convert_back_to_str(self, text: torch.Tensor):
        max_len = text.shape[-1]
        text = text.cpu().numpy().astype(np.uint8)
        return convert_byte_to_string(text[0], max_len)

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        if self.cfg.text_input_uuid in obs:
            obs[self.cfg.text_input_uuid] = self.convert_back_to_str(obs[self.cfg.text_input_uuid])
        obs_to_process = {k: obs[k] for k in self.input_uuids}
        processed_obs = self.preprocessor.process(
            [{"input_sensors": obs_to_process}], compute_image_feature_once=True
        )
        for uuid in self.cfg.feats_uuid:
            if self.cfg.text_input_uuid in uuid:
                processed_obs[uuid] = processed_obs[uuid].unsqueeze(0)
        return {uuid: processed_obs[uuid].squeeze(0) for uuid in self.cfg.feats_uuid}


@dataclass
class RLImageAugmentationPreprocessorConfig(SigLipPreprocessorConfig):
    rgb_input_uuid: List[str] = field(default_factory=lambda: [])
    output_uuid: str = ""
    feats_uuid: List[str] = field(default_factory=lambda: [])

    @property
    def output_shape(self):
        return self.image_encoder.value[1].output_size


class RLImageAugmentationPreprocessor(AllenActPreprocessor):
    def __init__(
        self,
        cfg: RLImageAugmentationPreprocessorConfig,
    ):
        self.cfg = cfg
        self.feats_uuid = cfg.feats_uuid

        self.device = None

        self.img_enc_cls, self.img_enc_cfg = cfg.image_encoder.value
        self._image_encoder = None

        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=cfg.output_shape)

        super().__init__(
            input_uuids=cfg.rgb_input_uuid,
            output_uuid=cfg.output_uuid,
            observation_space=observation_space,
        )

    @property
    def image_encoder(self):
        if self._image_encoder is None:
            self._image_encoder = self.img_enc_cls(self.img_enc_cfg)
            self._image_encoder.eval()
        return self._image_encoder

    @property
    def augmentations(self):
        return tensor_image_preprocessor(
            size=self.cfg.image_size,
            data_augmentation=True,
            specific=False,
            augmentation_version="v2",
            mean=self.cfg.mean,
            std=self.cfg.stdev,
            padding=False
        )
    
    
    def process_frames(self, obs):
        # mask the second camera if locoot
        if DEFAULT_ASSET in ['locobot', 'unitree_a1'] and 'raw_manipulation_camera' in obs.keys():
            obs['raw_manipulation_camera'] = torch.zeros_like(obs['raw_manipulation_camera'])

        
        return obs

    def to(self, device: torch.device) -> "RLImageAugmentationPreprocessor":
        self._image_encoder = self.image_encoder.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        
        
        obs = self.process_frames(obs, drop_manip_camera_indices)
        obs = self.process_agent_parameter(obs, drop_manip_camera_indices)
        x = torch.cat([obs[uuid].permute(0, 3, 1, 2) for uuid in self.input_uuids], dim=0).to(
            self.device
        )
        x = self.augmentations(x)
        x = self.image_encoder(x).clone().float()
        b = x.shape[0]
        sub_b = b // len(self.input_uuids)
        o = {uuid: x[idx * sub_b : (idx + 1) * sub_b] for idx, uuid in enumerate(self.feats_uuid)}
        return o


@dataclass
class RLTextPreprocessorConfig:
    text_uuid: str
    output_uuid: str
    output_shape: Tuple[int, int] = (4, 384)
    text_encoder_context_length: int = 64
    text_encoder: TEXT_ENCODER = TEXT_ENCODER.SigLIPBase
    goal_text_padding_length: int = 0

    @property
    def model_version(self) -> str:
        if "t5" in self.text_encoder.value[1].model_name:
            return self.text_encoder.value[1].model_name
        else:
            return f"hf-hub:timm/{self.text_encoder.value[1].model_name}"


class RLTextPreprocessor(AllenActPreprocessor):
    def __init__(
        self,
        cfg: RLTextPreprocessorConfig,
    ):
        self.cfg = cfg
        self.text_enc_cls, self.text_enc_cfg = cfg.text_encoder.value
        self._text_encoder = None
        self.device = None

        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=cfg.output_shape)

        super().__init__(
            input_uuids=[cfg.text_uuid],
            output_uuid=cfg.output_uuid,
            observation_space=observation_space,
        )

    @property
    def text_encoder(self):
        if self._text_encoder is None:
            self._text_encoder = self.text_enc_cls(self.text_enc_cfg)
            self._text_encoder.eval()
        return self._text_encoder

    def to(self, device: torch.device) -> "RLTextPreprocessor":
        self._text_encoder = self.text_encoder.to(device)
        self.device = device
        return self

    @lazy_property
    def text_preprocessor(self):
        if "t5" in self.cfg.model_version:
            return AutoTokenizer.from_pretrained("t5-small")
        else:
            return get_tokenizer(self.cfg.model_version)

    def process_goals(self, batch):
        if "t5" in self.cfg.model_version:
            goal_spec = self.text_preprocessor(
                [sample["goal"] for sample in batch],
                return_tensors="pt",
                padding=True,
            )
            if self.cfg.goal_text_padding_length > goal_spec.data["input_ids"].shape[1]:
                goal_spec.data["input_ids"] = torch.nn.functional.pad(
                    goal_spec.data["input_ids"],
                    (0, self.cfg.goal_text_padding_length - goal_spec.data["input_ids"].shape[1]),
                )
                goal_spec.data["attention_mask"] = torch.nn.functional.pad(
                    goal_spec.data["attention_mask"],
                    (
                        0,
                        self.cfg.goal_text_padding_length
                        - goal_spec.data["attention_mask"].shape[1],
                    ),
                    mode="constant",
                    value=1,
                )
            return {k: v.to(self.device) for k, v in goal_spec.items()}
        else:
            goal_spec = self.text_preprocessor(
                [sample["goal"] for sample in batch],
                context_length=self.cfg.text_encoder_context_length,  # for SigLIP
            )
            return goal_spec.to(self.device)

    def convert_back_to_str(self, text: torch.Tensor):
        max_len = text.shape[-1]
        goals = []
        text = text.cpu().numpy().astype(np.uint8)
        for g in text:
            g = convert_byte_to_string(g, max_len)
            goals.append({"goal": g})
        return goals

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        text_goal = self.process_goals(self.convert_back_to_str(obs[self.input_uuids[0]]))
        return self.text_encoder(text_goal).clone().float()