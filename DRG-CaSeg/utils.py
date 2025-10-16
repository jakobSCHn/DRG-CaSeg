import os
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Global random seed set to {seed}")