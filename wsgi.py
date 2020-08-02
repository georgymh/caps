import os
import logging

from utils.flask_utils import process_args, set_up_tensorflow
from app import create_app

logging.basicConfig(
   level=logging.DEBUG,
   format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
   datefmt='%Y-%m-%d %H:%M:%S',
   handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

FROZEN_MODEL_FILENAME = os.getenv(
    "FROZEN_MODEL_FILENAME",
    "models/gramcapsv0_optimized.pb"
)
logger.info("Using the model at {}".format(FROZEN_MODEL_FILENAME))

CONFIG_FILENAME = os.getenv(
    "CONFIG_FILENAME",
    "models/gramcapsv0_config.json"
)
logger.info("Using the config at {}".format(CONFIG_FILENAME))

GPU_MEMORY = float(os.getenv(
    "GPU_MEMORY",
    "0.0"
))
logger.info("Using the gpu memory fraction: {}".format(GPU_MEMORY))

predict_caption_fn, postprocess_caption_fn = set_up_tensorflow(
    FROZEN_MODEL_FILENAME,
    CONFIG_FILENAME,
    GPU_MEMORY,
)
app = create_app(predict_caption_fn, postprocess_caption_fn)
