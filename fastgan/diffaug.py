# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import tensorflow as tf


def DiffAugment(x, policy="", dtype=tf.float32, channels_first=False):
    if policy:
        if channels_first:
            x = tf.transpose(x, [0, 2, 3, 1])
        for p in policy.split(","):
            for f in AUGMENT_FNS[p]:
                x = f(x, dtype)
        if channels_first:
            x = tf.transpose(x, [0, 3, 1, 2])
    return x


@tf.function
def rand_brightness(x, dtype):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], dtype=dtype) - 0.5
    x = x + magnitude
    return x


@tf.function
def rand_saturation(x, dtype):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], dtype=dtype) * 2
    x_mean = tf.reduce_mean(x, axis=3, keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


@tf.function
def rand_contrast(x, dtype):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], dtype=dtype) + 0.5
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


@tf.function
def rand_translation(x, dtype, ratio=0.125):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    shift = tf.cast(image_size, dtype) * ratio + 0.5
    translation_x = tf.random.uniform([batch_size, 1], -shift[0], shift[0] + 1)
    translation_y = tf.random.uniform([batch_size, 1], -shift[1], shift[1] + 1)

    translation_x = tf.cast(translation_x, tf.int32)
    translation_y = tf.cast(translation_y, tf.int32)
    grid_x = tf.clip_by_value(
        tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) + translation_x + 1,
        0,
        image_size[0] + 1,
    )
    grid_y = tf.clip_by_value(
        tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) + translation_y + 1,
        0,
        image_size[1] + 1,
    )
    x = tf.gather_nd(
        tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]]),
        tf.expand_dims(grid_x, -1),
        batch_dims=1,
    )
    x = tf.transpose(
        tf.gather_nd(
            tf.pad(tf.transpose(x, [0, 2, 1, 3]), [[0, 0], [1, 1], [0, 0], [0, 0]]),
            tf.expand_dims(grid_y, -1),
            batch_dims=1,
        ),
        [0, 2, 1, 3],
    )
    return x


@tf.function
def rand_cutout(x, dtype, ratio=0.5):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    cutout_size = tf.cast(tf.cast(image_size, dtype) * ratio + 0.5, tf.int32)
    offset_x = tf.random.uniform(
        [tf.shape(x)[0], 1, 1],
        maxval=tf.cast(image_size[0] + (1 - cutout_size[0] % 2), dtype),
    )
    offset_y = tf.random.uniform(
        [tf.shape(x)[0], 1, 1],
        maxval=tf.cast(image_size[1] + (1 - cutout_size[1] % 2), dtype),
    )
    offset_x = tf.cast(offset_x, tf.int32)
    offset_y = tf.cast(offset_y, tf.int32)

    grid_batch, grid_x, grid_y = tf.meshgrid(
        tf.range(batch_size, dtype=tf.int32),
        tf.range(cutout_size[0], dtype=tf.int32),
        tf.range(cutout_size[1], dtype=tf.int32),
        indexing="ij",
    )
    cutout_grid = tf.stack(
        [
            grid_batch,
            grid_x + offset_x - cutout_size[0] // 2,
            grid_y + offset_y - cutout_size[1] // 2,
        ],
        axis=-1,
    )
    cutout_grid = tf.maximum(cutout_grid, 0)
    cutout_grid = tf.minimum(cutout_grid, tf.reshape(tf.shape(x)[:3] - 1, [1, 1, 1, 3]))
    mask = tf.maximum(
        1
        - tf.tensor_scatter_nd_add(
            tf.zeros(tf.shape(x)[:3], dtype),
            cutout_grid,
            tf.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=dtype),
        ),
        0,
    )
    x = x * tf.expand_dims(mask, axis=3)
    return x


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}
