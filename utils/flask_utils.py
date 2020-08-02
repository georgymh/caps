import argparse
import time

from utils.model_utils import (
    load_graph,
    load_postprocessing_config,
    get_input_and_output_tensors,
    run_session,
    model_output_to_captions,
)

import cv2
import numpy as np
import tensorflow as tf


class Cronometer:
    def __init__(self):
        self._times = [time.time()]

    def record_time(self, from_start=False):
        current_time = time.time()
        past_time = self._times[0] if from_start else self._times[-1]
        self._times.append(current_time)
        return current_time - past_time


def read_image_from_request(request):
    image_str = request.files['image'].read()
    data_array = np.fromstring(image_str, np.uint8)
    image = cv2.imdecode(data_array, cv2.IMREAD_COLOR)
    return image


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen_model_filename",
        default="models/gramcapsv0_optimized.pb",
        type=str,
        help="Frozen model file to import"
    )
    parser.add_argument(
        "--config_filename",
        default="models/gramcapsv0_config.json",
        type=str,
        help="Config to conver the captions"
    )
    parser.add_argument(
        "--gpu_memory",
        default=0.0,
        type=float,
        help="GPU memory per process"
    )
    args = parser.parse_args()
    return args


def set_up_tensorflow(frozen_model_filename, config_filename, gpu_memory):
    """
    Sets up the Tensorflow config and returns two objects:
        - a prediction function, that runs an image through the model
        - a post-processing function, that converts the output of the model into a caption
    """
    print("Loading the model")
    graph = load_graph(frozen_model_filename)
    x, y = get_input_and_output_tensors(graph, auto_infer_tensor_names=True)

    print("Starting Session, setting the GPU memory usage to %f" % gpu_memory)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)
    sess_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    persistent_sess = tf.Session(graph=graph, config=sess_config)

    print("Preparing forward-pass function")
    predict_caption_fn = lambda image: run_session(persistent_sess, x, y, image)

    print("Loading post-processing config")
    # TODO: Change this to be a JSON load instead; not pickle.
    captions_config = load_postprocessing_config(config_filename)
    postprocess_caption_fn = lambda output: model_output_to_captions(output, captions_config)

    print("Returning prediction and post-processing functions")
    return predict_caption_fn, postprocess_caption_fn
