from types import FunctionType
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torchvision.transforms._functional_video import normalize

from flash import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _PYTORCHVIDEO_AVAILABLE

_VIDEO_DETECTION_BACKBONES = FlashRegistry("backbones")
_IMAGE_DETECTOR_BACKBONES = FlashRegistry("backbones")

if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.data.encoded_video import EncodedVideo
    from pytorchvideo.models import hub
    from pytorchvideo.transforms.functional import (
        clip_boxes_to_image,
        short_side_scale_with_boxes,
        uniform_temporal_subsample,
    )

    for fn_name in dir(hub):
        if "__" not in fn_name:
            fn = getattr(hub, fn_name)
            if isinstance(fn, FunctionType):
                _VIDEO_DETECTION_BACKBONES(fn=fn, name=fn_name)


class VideoObjectDetector(Task):
    backbones: FlashRegistry = _VIDEO_DETECTION_BACKBONES
    image_detector_backbones: FlashRegistry = _IMAGE_DETECTOR_BACKBONES

    required_extras = "video"

    def __init__(
        self,
        model: Union[nn.Module, str] = "slow_r50",
        num_frames: int = 4,
        pretrained: bool = True,
        image_detector: Union[nn.Module, str] = "faster_rcnn_R_50_FPN_3x",
        image_detector_kwargs={},
        filter_boxes_fn: Optional[Callable] = None,
    ) -> None:

        if isinstance(model, str):
            self.model: nn.Module = self.backbones.get(model)(pretrained=pretrained)
        else:
            self.model = model

        if isinstance(image_detector, str):
            self.image_detector: nn.Module = self.image_detector_backbones.get(image_detector)(**image_detector_kwargs)
        else:
            self.image_detector = image_detector

        self.num_frames = num_frames
        self.filter_boxes_fn = filter_boxes_fn
        super().__init__(model=self.model, )

    def prepare_data(self, x) -> EncodedVideo:
        encoded_vid = EncodedVideo.from_path(x)
        return encoded_vid

    def _extract_clip(self, x: EncodedVideo, time_stamp: int, clip_duration: float = 1.0):
        inp_imgs = x.get_clip(
            time_stamp - clip_duration / 2.0,  # start second
            time_stamp + clip_duration / 2.0,  # end second
        )
        inp_imgs = inp_imgs["video"]

        inp_img = inp_imgs[:, inp_imgs.shape[1] // 2, :, :]
        inp_img = inp_img.permute(1, 2, 0)
        return inp_img

    def image_detector_forward(self, x: Any):
        predicted_boxes = self.image_detector(x)
        if self.filter_boxes_fn:
            predicted_boxes = self.filter_boxes_fn(predicted_boxes)
        return predicted_boxes

    def prepare_video_forward(self, inp_imgs: torch.Tensor, predicted_boxes: torch.Tensor):
        inputs, inp_boxes, _ = self.ava_inference_transform(inp_imgs, predicted_boxes.numpy())

        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)

        if isinstance(inputs, list):
            inputs = [inp.unsqueeze(0) for inp in inputs]
        else:
            inputs = inputs.unsqueeze(0)

        return inputs, inp_boxes

    def forward(
        self,
        x: Any,
        time_stamp_range: Optional[Tuple[int, int]] = None,
        clip_duration: float = 1.0,
    ) -> Any:
        data = self.prepare_data(x)

        if time_stamp_range:
            time_stamp_range = range(*time_stamp_range)
        else:
            time_stamp_range = range(int(data.duration))

        for time_stamp in time_stamp_range:
            inp_img = self._extract_clip(data, time_stamp, clip_duration)
            predicted_boxes = self.image_detector_forward(inp_img)

            if len(predicted_boxes) == 0:
                continue

            inputs, inp_boxes = self.prepare_video_forward(inp_img, predicted_boxes)
            outputs = self.model(inputs, inp_boxes)
            yield outputs

    @staticmethod
    def ava_inference_transform(
        clip: torch.Tensor,
        boxes: List,
        num_frames=4,
        crop_size=256,
        data_mean=(0.45, 0.45, 0.45),
        data_std=(0.225, 0.225, 0.225),
        slow_fast_alpha=None,
    ):

        boxes = np.array(boxes)
        ori_boxes = boxes.copy()

        clip = uniform_temporal_subsample(clip, num_frames)
        clip = clip.float()
        clip = clip / 255.0

        height, width = clip.shape[2], clip.shape[3]
        boxes = clip_boxes_to_image(boxes, height, width)

        clip, boxes = short_side_scale_with_boxes(
            clip,
            size=crop_size,
            boxes=boxes,
        )

        clip = normalize(
            clip,
            np.array(data_mean, dtype=np.float32),
            np.array(data_std, dtype=np.float32),
        )

        boxes = clip_boxes_to_image(boxes, clip.shape[2], clip.shape[3])

        # Incase of slowfast, generate both pathways
        if slow_fast_alpha is not None:
            fast_pathway = clip
            # Perform temporal sampling from the fast pathway.
            slow_pathway = torch.index_select(
                clip,
                1,
                torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long(),
            )
            clip = [slow_pathway, fast_pathway]

        return clip, torch.from_numpy(boxes), ori_boxes
