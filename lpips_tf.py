import os
import sys

import tensorflow as tf
import urllib

_URL = "http://rail.eecs.berkeley.edu/models/lpips"


def _download(url, output_dir):
    """Downloads the `url` file into `output_dir`.

    Modified from https://github.com/tensorflow/models/blob/master/research/slim/datasets/dataset_utils.py
    """
    filename = url.split("/")[-1]
    filepath = os.path.join(output_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write(
            "\r>> Downloading %s %.1f%%"
            % (filename, float(count * block_size) / float(total_size) * 100.0)
        )
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print("Successfully downloaded", filename, statinfo.st_size, "bytes.")


class PerceptualLoss:
    def __init__(self, model="net-lin", net="alex", version=0.1):
        cache_dir = os.path.expanduser("~/.lpips")
        os.makedirs(cache_dir, exist_ok=True)
        pb_fname = f"{model}_{net}_v{version}.pb"

        if not os.path.isfile(os.path.join(cache_dir, pb_fname)):
            _download(_URL + "/" + pb_fname, cache_dir)

        with open(os.path.join(cache_dir, pb_fname), "rb") as f:
            self.graph_def = tf.compat.v1.GraphDef()
            self.graph_def.ParseFromString(f.read())

    @tf.function
    def __call__(self, input0, input1):
        """
        Learned Perceptual Image Patch Similarity (LPIPS) metric.

        Args:
            input0: An image tensor of shape `[..., height, width, channels]`,
                with values in [0, 1].
            input1: An image tensor of shape `[..., height, width, channels]`,
                with values in [0, 1].

        Returns:
            The Learned Perceptual Image Patch Similarity (LPIPS) distance.

        Reference:
            Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang.
            The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
            In CVPR, 2018.
        """

        # flatten the leading dimensions
        batch_shape = tf.shape(input0)[:-3]
        input0 = tf.reshape(input0, tf.concat([[-1], tf.shape(input0)[-3:]], axis=0))
        input1 = tf.reshape(input1, tf.concat([[-1], tf.shape(input1)[-3:]], axis=0))
        # NHWC to NCHW
        input0 = tf.transpose(input0, [0, 3, 1, 2])
        input1 = tf.transpose(input1, [0, 3, 1, 2])
        # normalize to [-1, 1]
        input0 = input0 * 2.0 - 1.0
        input1 = input1 * 2.0 - 1.0

        _ = tf.compat.v1.import_graph_def(
            self.graph_def, input_map={"0:0": input0, "1:0": input1}
        )
        (distance,) = tf.compat.v1.get_default_graph().get_operations()[-1].outputs

        if distance.shape.ndims == 4:
            distance = tf.squeeze(distance, axis=[-3, -2, -1])
        # reshape the leading dimensions
        distance = tf.reshape(distance, batch_shape)
        return distance


# @tf.function
# def lpips(input0, input1, model="net-lin", net="alex", version=0.1):
#     """
#     Learned Perceptual Image Patch Similarity (LPIPS) metric.

#     Args:
#         input0: An image tensor of shape `[..., height, width, channels]`,
#             with values in [0, 1].
#         input1: An image tensor of shape `[..., height, width, channels]`,
#             with values in [0, 1].

#     Returns:
#         The Learned Perceptual Image Patch Similarity (LPIPS) distance.

#     Reference:
#         Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang.
#         The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
#         In CVPR, 2018.
#     """

#     # flatten the leading dimensions
#     batch_shape = tf.shape(input0)[:-3]
#     input0 = tf.reshape(input0, tf.concat([[-1], tf.shape(input0)[-3:]], axis=0))
#     input1 = tf.reshape(input1, tf.concat([[-1], tf.shape(input1)[-3:]], axis=0))
#     # NHWC to NCHW
#     input0 = tf.transpose(input0, [0, 3, 1, 2])
#     input1 = tf.transpose(input1, [0, 3, 1, 2])
#     # normalize to [-1, 1]
#     input0 = input0 * 2.0 - 1.0
#     input1 = input1 * 2.0 - 1.0

#     input0_name, input1_name = "0:0", "1:0"

#     default_graph = tf.compat.v1.get_default_graph()

#     cache_dir = os.path.expanduser("~/.lpips")
#     os.makedirs(cache_dir, exist_ok=True)
#     pb_fname = f"{model}_{net}_v{version}.pb"

#     if not os.path.isfile(os.path.join(cache_dir, pb_fname)):
#         _download(_URL + "/" + pb_fname, cache_dir)

#     with open(os.path.join(cache_dir, pb_fname), "rb") as f:
#         graph_def = tf.compat.v1.GraphDef()
#         graph_def.ParseFromString(f.read())
#         _ = tf.compat.v1.import_graph_def(
#             graph_def, input_map={input0_name: input0, input1_name: input1}
#         )
#         (distance,) = tf.compat.v1.get_default_graph().get_operations()[-1].outputs

#     if distance.shape.ndims == 4:
#         distance = tf.squeeze(distance, axis=[-3, -2, -1])
#     # reshape the leading dimensions
#     distance = tf.reshape(distance, batch_shape)
#     return distance
