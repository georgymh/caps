# NOTE: Needed for Pickle. (Needs to be able to `import configuration`.)
import os, sys, pickle
sys.path.insert(0, os.path.abspath('./utils/3p'))
########################################################################

import unicodedata

import tensorflow as tf
# NOTE: TO PREVENT THE 'GatherTree' ERROR WE NEED TO DO THIS.
# MORE INFO: https://github.com/tensorflow/tensorflow/issues/12927
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def get_input_and_output_tensors(
    graph,
    auto_infer_tensor_names = True,
    input_tensor_name = None,
    output_tensor_name = None
):
    assert auto_infer_tensor_names or (input_tensor_name and output_tensor_name)
    if auto_infer_tensor_names:
        all_ops = [op.name for op in graph.get_operations()]
        input_tensor_name = all_ops[0] + ":0"
        output_tensor_name = all_ops[-1] + ":0"
    x_tensor = graph.get_tensor_by_name(input_tensor_name)
    y_tensor = graph.get_tensor_by_name(output_tensor_name)
    return x_tensor, y_tensor


def run_graph_once(graph, x_tensor, y_tensor, graph_input):
    with tf.Session(graph=graph) as sess:
        out = run_session(
            sess=sess,
            x_tensor=x_tensor,
            y_tensor=y_tensor,
            graph_input=graph_input,
        )
    return out


def run_session(sess, x_tensor, y_tensor, graph_input):
    # Note: We don't need to initialize/restore anything.
    # There is no Variables in this graph, only hardcoded constants :)
    y_out = sess.run(y_tensor, feed_dict={
        x_tensor: graph_input
    })
    return y_out


def load_postprocessing_config(config_filename):
    # NOTE: Needed for Pickle. (Needs to be able to `import configuration`.)
    import os, sys
    sys.path.insert(0, os.path.abspath('../utils/3p'))
    with open(config_filename, "r") as f:
        config_obj = pickle.load(f)
    return config_obj


def model_output_to_captions(ids, config):
    """
    NOTE: Need to refactor this code ASAP.
    """
    def _number_to_base(n, base):
        """Function to convert any base-10 integer to base-N."""
        if base < 2:
            raise ValueError('Base cannot be less than 2.')
        if n < 0:
            sign = -1
            n *= sign
        elif n == 0:
            return [0]
        else:
            sign = 1
        digits = []
        while n:
            digits.append(sign * int(n % base))
            n //= base
        return digits[::-1]

    def _baseN_arr_to_dec(baseN_array, base):
        """Convert base-N array / list to base-10 number."""
        result = 0
        power = len(baseN_array) - 1
        for num in baseN_array:
            result += num * pow(base, power)
            power -= 1
        return result

    captions = []
    if config.token_type == 'radix':
        # Convert Radix IDs to sentence.
        base = config.radix_base
        vocab_size = len(config.itow)
        word_len = len(_number_to_base(vocab_size, base))
        for i in range(ids.shape[0]):
            sent = []
            row = [wid for wid in ids[i, :] if wid < base and wid >= 0]
            if len(row) % word_len != 0:
                row = row[:-1]
            for j in range(0, len(row), word_len):
                word_id = _baseN_arr_to_dec(row[j:j + word_len], base)
                if word_id < vocab_size:
                    sent.append(config.itow[str(word_id)])
                else:
                    pass
            captions.append(' '.join(sent))
    else:
        # Convert word / char IDs to sentence.
        for i in range(ids.shape[0]):
            row = [wid for wid in ids[i, :]
                    if wid >= 0 and wid != config.wtoi['<EOS>']]
            sent = [config.itow[str(w)] for w in row]
            if config.token_type == 'word':
                captions.append(' '.join(sent))
            elif config.token_type == 'char':
                captions.append(''.join(sent))

    if len(captions) == 1:
        caption = captions[0]
        return unicodedata.normalize('NFKD', caption).encode('ascii','ignore')
    return captions
