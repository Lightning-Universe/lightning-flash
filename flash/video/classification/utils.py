from typing import List, Optional, Tuple, Type

import torch

from flash.core.utilities.imports import _VIDEO_AVAILABLE

if _VIDEO_AVAILABLE:
    from pytorchvideo.data.utils import MultiProcessSampler
else:
    MultiProcessSampler = None


class LabeledVideoTensorDataset(torch.utils.data.IterableDataset):
    """LabeledVideoTensorDataset handles a direct tensor input data."""

    def __init__(
        self,
        labeled_video_tensors: List[Tuple[str, Optional[dict]]],
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    ) -> None:
        self._labeled_videos = labeled_video_tensors

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(self._labeled_videos, generator=self._video_random_generator)
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None

    def __next__(self) -> dict:
        """Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        # Reuse previously stored video if there are still clips to be sampled from
        # the last loaded video.
        video_index = next(self._video_sampler_iter)
        video_tensor, info_dict = self._labeled_videos[video_index]
        self._loaded_video_label = (video_tensor, info_dict, video_index)

        sample_dict = {
            "video": self._loaded_video_label[0],
            "video_name": f"video{video_index}",
            "video_index": video_index,
            "label": info_dict,
            "video_label": info_dict,
        }

        return sample_dict

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self

    def size(self):
        return len(self._labeled_videos)
