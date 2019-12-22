# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 12:18:33
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:33:11



import os
import json
import glob
from datetime import datetime
from pathlib import Path

import random
from functools import partial
import pandas as pd
import cv2
import numpy as np
from torch.nn.parallel.data_parallel import data_parallel
import torch
import math
import tqdm

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def set_seed(seed=6750):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True

set_seed(1217)

ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ
DATA_ROOT = Path('../input/understanding-cloud-organization' if ON_KAGGLE else '../input/')

PIXEL_THRESHOLDS = [0.5, 0.5, 0.5, 0.5]
AREA_SIZES = [0, 0, 0, 0]
CLASS_NAMES = ['Fish', 'Flower', 'Gravel', 'Sugar']
NUM_CLASSES = 4
SIZE = (384, 576) #h, w, raw = (1400x2100)
TEST_SIZE = (384, 576)
NFOLDS = 5
