import os
import pickle
import argparse

from utils.model_utils import (
    load_graph,
    load_postprocessing_config,
    get_input_and_output_tensors,
    run_graph_once,
    model_output_to_captions,
)

import cv2


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# TEST GRAMCAPS:
#   - python execute_model.py -m models/gramcapsv0_optimized.pb -a --input_image assets/test_image.jpg --show_caption


def make_args():
    parser = argparse.ArgumentParser()

    default_frozen_model = os.path.join(CURRENT_DIR, "assets", "test_model.pb")
    parser.add_argument(
        "-m",
        "--frozen_model_filename",
        default=default_frozen_model,
        type=str,
        help="Frozen model file to import"
    )

    parser.add_argument(
        "-i",
        "--input_tensor_name",
        default="prefix/I:0",
        type=str,
        help="Input tensor name"
    )

    parser.add_argument(
        "-o",
        "--output_tensor_name",
        default="prefix/O:0",
        type=str,
        help="Output tensor name"
    )

    parser.add_argument(
        "-x",
        "--input",
        default=[1, 1, 1],
        nargs='+',
        type=int,
        help="Input to the model"
    )

    parser.add_argument(
        "--input_image",
        default="",
        type=str,
        help="Input image to the model"
    )

    parser.add_argument(
        "-a",
        "--auto_infer_tensor_names",
        default=True,
        action="store_true",
        help="Whether to auto infer input and outpu tensor names"
    )
    parser.add_argument(
        "--show_caption",
        default=False,
        action="store_true",
        help="Show the caption from the model's output"
    )

    args = parser.parse_args()

    if args.input_image:
        args.input = cv2.imread(args.input_image)
        print("Loaded image from '{}' with size '{}'".format(args.input_image, args.input.shape))
    return args


def print_graph_ops(graph):
    print("Printing ops in the graph...")
    for op in graph.get_operations():
        print("\t {}".format(op.name))
    print("\n\n")


if __name__ == '__main__':
    args = make_args()

    graph = load_graph(args.frozen_model_filename)
    # print_graph_ops(graph)

    x_tensor, y_tensor = get_input_and_output_tensors(
        graph,
        args.auto_infer_tensor_names,
        args.input_tensor_name,
        args.output_tensor_name
    )

    output = run_graph_once(graph, x_tensor, y_tensor, args.input)

    print("Input to the model:\n\t{}\n".format(args.input))
    print("Output of the model:\n\t{}\n".format(output))

    if args.show_caption:
        # NOTE: Do not hard-code this in the future.
        config_path = os.path.join(CURRENT_DIR, "models", "gramcapsv0_config.json")
        config_obj = load_postprocessing_config(config_path)
        caption = model_output_to_captions(output, config_obj)
        print("Caption of the model:\n\t{}".format(caption))
