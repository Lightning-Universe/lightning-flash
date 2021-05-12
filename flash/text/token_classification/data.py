import os
from functools import partial
from itertools import chain, groupby
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from datasets import DatasetDict, load_dataset
from tokenizers.pre_tokenizers import WhitespaceSplit
from torch import Tensor
from transformers import AutoTokenizer, default_data_collator
from transformers.modeling_outputs import TokenClassifierOutput

from flash.data.auto_dataset import AutoDataset
from flash.data.data_module import DataModule
from flash.data.data_source import DataSource, DefaultDataSources, LabelsState
from flash.data.process import Postprocess, Preprocess

LABEL_IGNORE = -100


class TokenDataSource(DataSource):

    def __init__(self, backbone: str, max_length: int = 128):
        super().__init__()

        self.pre_tokenizer = WhitespaceSplit()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        self.max_length = max_length

    def _pre_tokenize_fn(self, ex: Dict[str, str], input: str):
        pre_tokenized = self.pre_tokenizer.pre_tokenize_str(ex[input])
        ex[input] = [chunk for (chunk, span) in pre_tokenized]

        return ex

    def _tokenize_fn(self, ex: Union[Dict[str, str], str], input: str):
        tokenized = self.tokenizer(
            ex[input],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            is_split_into_words=True,
        )
        tokenized["word_ids"] = [tokenized.word_ids(i) for i, _ in enumerate(tokenized.input_ids)]

        return tokenized

    def _align_labels_fn(
        self,
        ex: Dict[str, str],
        target: str,
        label_all_subword_tokens: bool = False,
    ):
        """Inspired by
        https://github.com/huggingface/transformers/blob/v4.5.1/examples/token-classification/run_ner.py#L318"""

        aligned_labels = []

        for i, (word_ids, labels) in enumerate(zip(ex["word_ids"], ex[target])):
            word_labels = labels.strip().split()
            token_labels = []

            for word_id, group in groupby(word_ids):
                token_cnt = len(list(group))
                # special token
                if word_id is None:
                    token_labels += [None] * token_cnt
                else:
                    label = word_labels[word_id]
                    if label_all_subword_tokens:
                        token_labels += [label] * token_cnt
                    else:
                        token_labels += [label] + ([None] * (token_cnt - 1))

            aligned_labels.append(token_labels)

        ex[target] = aligned_labels

        return ex

    def _transform_labels(
        self,
        label_to_class_mapping: Dict[str, int],
        target: str,
        ex: Dict[str, Union[int, str]],
    ):
        ex[target] = [LABEL_IGNORE if lbl is None else label_to_class_mapping[lbl] for lbl in ex[target]]

        return ex


class TokenFileDataSource(TokenDataSource):

    def __init__(self, filetype: str, backbone: str, max_length: int = 128):
        super().__init__(backbone, max_length=max_length)

        self.filetype = filetype

    def load_data(
        self,
        data: Tuple[str, Union[str, List[str]], Union[str, List[str]]],
        dataset: Optional[Any] = None,
        columns: Union[List[str], Tuple[str]] = (
            "input_ids",
            "attention_mask",
            "labels",
        ),
        use_full: bool = True,
    ) -> Union[Sequence[Mapping[str, Any]]]:
        csv_file, input, target = data

        data_files = {}

        stage = self.running_stage.value
        data_files[stage] = str(csv_file)

        # FLASH_TESTING is set in the CI to run faster.
        if use_full and os.getenv("FLASH_TESTING", "0") == "0":
            dataset_dict = load_dataset(self.filetype, data_files=data_files)
        else:
            # used for debugging. Avoid processing the entire dataset   # noqa E265
            dataset_dict = DatasetDict({
                stage: load_dataset(self.filetype, data_files=data_files, split=[f"{stage}[:20]"])[0]
            })

        if self.training:
            labels = sorted(set(chain.from_iterable(map(str.split, dataset_dict["train"][target]))))
            dataset.num_classes = len(labels)
            self.set_state(LabelsState(labels))

        dataset_dict = dataset_dict.map(partial(self._pre_tokenize_fn, input=input))
        dataset_dict = dataset_dict.map(partial(self._tokenize_fn, input=input), batched=True)
        dataset_dict = dataset_dict.map(
            partial(self._align_labels_fn, target=target),
            batched=True,
        )

        labels = self.get_state(LabelsState)
        if labels is not None:
            labels = labels.labels
            label_to_class_mapping = {lbl: idx for idx, lbl in enumerate(labels)}
            dataset_dict = dataset_dict.map(partial(self._transform_labels, label_to_class_mapping, target))

        # Hugging Face models expect target to be named ``labels``.
        if not self.predicting and target != "labels":
            dataset_dict.rename_column_(target, "labels")

        dataset_dict.set_format("torch", columns=columns)

        return dataset_dict[stage]

    def predict_load_data(self, data: Any, dataset: AutoDataset):
        return self.load_data(data, dataset, columns=["input_ids", "attention_mask"])


class TokenCSVDataSource(TokenFileDataSource):

    def __init__(self, backbone: str, max_length: int = 128):
        super().__init__("csv", backbone, max_length=max_length)


class TokenJSONDataSource(TokenFileDataSource):

    def __init__(self, backbone: str, max_length: int = 128):
        super().__init__("json", backbone, max_length=max_length)


class TokenClassificationPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        backbone: str = "prajjwal1/bert-tiny",
        max_length: int = 128,
    ):
        self.backbone = backbone
        self.max_length = max_length

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.CSV: TokenCSVDataSource(self.backbone, max_length=max_length),
                DefaultDataSources.JSON: TokenJSONDataSource(self.backbone, max_length=max_length),
            },
            default_data_source=DefaultDataSources.CSV,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "backbone": self.backbone,
            "max_length": self.max_length,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return cls(**state_dict)

    def per_batch_transform(self, batch: Any) -> Any:
        if "labels" not in batch:
            # todo: understand why an extra dimension has been added.
            if batch["input_ids"].dim() == 3:
                batch["input_ids"] = batch["input_ids"].squeeze(0)
        return batch

    def collate(self, samples: Any) -> Tensor:
        """Override to convert a set of samples to a batch"""
        if isinstance(samples, dict):
            samples = [samples]
        return default_data_collator(samples)


class TokenClassificationPostprocess(Postprocess):

    def per_batch_transform(self, batch: Any) -> Any:
        if isinstance(batch, TokenClassifierOutput):
            batch = batch.logits
        return super().per_batch_transform(batch)


class TokenClassificationData(DataModule):
    """Data Module for token classification tasks"""

    preprocess_cls = TokenClassificationPreprocess
    postprocess_cls = TokenClassificationPostprocess
