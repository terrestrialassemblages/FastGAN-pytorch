import os
import shutil
import json

import tensorflow as tf


def get_dir(args):
    task_name = "train_results/" + args.name
    saved_model_folder = os.path.join(task_name, "models")
    saved_image_folder = os.path.join(task_name, "images")
    log_folder = os.path.join(task_name, "logs")

    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)

    for f in os.listdir("./"):
        if ".py" in f:
            shutil.copy(f, task_name + "/" + f)

    with open(os.path.join(saved_model_folder, "../args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    return saved_model_folder, saved_image_folder, log_folder


def imgrid(imarray, cols=4):
    """Lays out a [N, H, W, C] image array as a single image grid."""
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray.shape
    rows = N // cols + int(N % cols != 0)
    grid = tf.reshape(
        tf.transpose(tf.reshape(imarray, [rows, cols, H, W, C]), [0, 2, 1, 3, 4]),
        [rows * H, cols * W, C],
    )
    return grid
