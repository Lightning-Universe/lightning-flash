from typing import Callable, Dict, Optional, Union

from flash.text.seq2seq.core.data import Seq2SeqData, Seq2SeqPostprocess, Seq2SeqPreprocess


class SentenceEmbedPreprocess(Seq2SeqPreprocess):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        backbone: str = "all-MiniLM-L6-v2",
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
        **kwargs,
    ):
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            backbone=backbone,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            **kwargs,
        )


class SentenceEmbedData(Seq2SeqData):

    preprocess_cls = SentenceEmbedPreprocess
    postprocess_cls = Seq2SeqPostprocess
