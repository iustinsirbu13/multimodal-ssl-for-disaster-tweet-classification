#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from models.bert import BertClf
from models.bow import GloveBowClf
from models.concat_bert import MultimodalConcatBertClf
from models.concat_bow import  MultimodalConcatBowClf
from models.image import ImageClf
from models.mmbt import MultimodalBertClf


MODELS = {
    "bert": BertClf,
    "bow": GloveBowClf,
    "concatbow": MultimodalConcatBowClf,
    "concatbert": MultimodalConcatBertClf,
    "img": ImageClf,
    "mmbt": MultimodalBertClf,
}


def get_model(args):
    return MODELS[args.model](args)
