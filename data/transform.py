from logging import getLogger
import random
import torch
import torch.nn as nn
from recbole.data.interaction import Interaction


def construct_transform(config):
    if config['transform'] is None:
        logger = getLogger()
        logger.warning('Equal transform')
        return Equal(config)


class Equal:
    def __init__(self, config):
        pass

    def __call__(self, dataloader, interaction):
        return interaction

