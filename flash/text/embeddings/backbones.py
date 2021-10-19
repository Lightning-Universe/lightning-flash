#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 12:13:39 2021

@author: abhijithneilabraham
"""
from flash.core.registry import ExternalRegistry, FlashRegistry
from flash.core.utilities.imports import _TEXT_AVAILABLE
from flash.core.utilities.providers import _SENTENCE_TRANSFORMERS

SENTENCE_TRANSFORMERS_BACKBONE = FlashRegistry("backbones")

if _TEXT_AVAILABLE:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_BACKBONE += ExternalRegistry(
        SentenceTransformer,
        "backbones",
        _SENTENCE_TRANSFORMERS,
    )

