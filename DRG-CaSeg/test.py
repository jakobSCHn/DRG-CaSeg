import os
import logging

import config
from utils import seed_everything

config.setup_logging()
logger = logging.getLogger(__name__)

import caiman as cm
from data_utils.wrangler import load_czi_to_caiman



if __name__ == "__main__":

    SEED = int(os.getenv("GLOBAL_SEED", 42))
    seed_everything(SEED)

    p = "/home/jaschneider/projects/DRG-CaSeg/data/test_00_3rdB2.czi"
    movie = load_czi_to_caiman(p)
    print(type(movie))
    print("Hello")