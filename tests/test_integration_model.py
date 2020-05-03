import os

from utils.model_utils import (
    load_graph,
    load_postprocessing_config,
    get_input_and_output_tensors,
    run_graph_once,
    model_output_to_captions,
)

import cv2


PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)


def test_integration_model():
    FROZEN_MODEL_FILENAME = os.path.join(PROJECT_DIR, "models", "gramcapsv0_optimized.pb")
    CONFIG_FILENAME = os.path.join(PROJECT_DIR, "models", "gramcapsv0_config.json")
    TEST_IMAGE_FILENAME = os.path.join(PROJECT_DIR, "assets", "test_image.jpg")

    img_input = cv2.imread(TEST_IMAGE_FILENAME)

    graph = load_graph(FROZEN_MODEL_FILENAME)
    x_tensor, y_tensor = get_input_and_output_tensors(graph, True)
    output = run_graph_once(graph, x_tensor, y_tensor, img_input)

    config_obj = load_postprocessing_config(CONFIG_FILENAME)
    caption = model_output_to_captions(output, config_obj)

    assert isinstance(caption, str), "Type of `caption` is {} not str.".format(type(caption))
    assert len(caption) > 3, "`caption` is too short... this is not an error but it's weird."
