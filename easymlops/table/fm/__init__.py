# -*- coding: utf-8 -*-
from easymlops.table.fm.base import FMBase
from easymlops.table.fm.fm import FMClassification, FMRegression
from easymlops.table.fm.ffm import FFMClassification, FFMRegression
from easymlops.table.fm.deepfm import DeepFMClassification, DeepFMRegression


__all__ = [
    "FMBase",
    "FMClassification",
    "FMRegression",
    "FFMClassification",
    "FFMRegression",
    "DeepFMClassification",
    "DeepFMRegression",
]
