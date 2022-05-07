import argparse
import glob
import os

import tensorflow as tf
from tqdm import tqdm


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, num_channels):
    def preprocess_image(image_string):
        image = tf.io.decode_png(image_string, channels=num_channels)
        image = tf.cast(image, tf.float16) - 127.5
        image = image / 127.5
        return tf.io.serialize_tensor(image)

    feature = {
        "image_raw": _bytes_feature(image_string),
        # "image_processed": _bytes_feature(preprocess_image(image_string)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(args):
    fnames = glob.glob(os.path.join(args.path, "*"))
    with tf.io.TFRecordWriter(args.name) as writer:
        for filename in tqdm(fnames):
            with open(filename, "rb") as f:
                data = image_example(f.read(), args.channels)
            writer.write(data.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--path", type=str, default="../apebase/ipfs", help="path of resource dataset"
    )
    parser.add_argument(
        "--name", type=str, default="apebase.tfrecords", help="output name"
    )
    parser.add_argument("--channels", type=int, default=3, help="number of channels")
    args = parser.parse_args()
    main(args)
