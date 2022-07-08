"""
TRAIN SKIP-ATTENTION GANOMALY

. Example: Run the following command from the terminal.
    run train.py                                    \
        --model <skipganomaly, ganomaly>            \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \
"""

##
# LIBRARIES

from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import glob
import os


def main():
    """ Training
    """
    # Arguments
    opt = Options().parse()

    # Load Data
    data = load_data(opt)
    # Load Model
    model = load_model(opt, data)
    # Train Model
    model.train()


if __name__ == '__main__':
    main()
