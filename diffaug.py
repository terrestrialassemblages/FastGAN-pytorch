# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

# import torch
# import torch.nn.functional as F
import tensorflow as tf


def DiffAugment(x, policy="", channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(","):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (tf.random.uniform([x.shape[0], 1, 1, 1], dtype=x.dtype) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (
        tf.random.uniform([x.shape[0], 1, 1, 1], dtype=x.dtype) * 2
    ) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (
        tf.random.uniform([x.shape[0], 1, 1, 1], dtype=x.dtype) + 0.5
    ) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.shape[2] * ratio + 0.5), int(x.shape[3] * ratio + 0.5)
    translation_x = tf.random.uniform(
        [x.shape[0], 1, 1], minval=-shift_x, maxval=shift_x + 1, dtype=tf.dtypes.int32
    )
    translation_y = tf.random.uniform(
        [x.shape[0], 1, 1], minval=-shift_y, maxval=shift_y + 1, dtype=tf.dtypes.int32
    )
    grid_batch, grid_x, grid_y = tf.meshgrid(
        tf.range(x.shape[0], dtype=tf.dtypes.int64),
        tf.range(x.shape[2], dtype=tf.dtypes.int64),
        tf.range(x.shape[3], dtype=tf.dtypes.int64),
    )
    grid_x = tf.clip_by_value(grid_x + translation_x + 1, 0, x.shape[2] + 1)
    grid_y = tf.clip_by_value(grid_y + translation_y + 1, 0, x.shape[3] + 1)
    x_pad = tf.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = (
        x_pad.permute(0, 2, 3, 1)
        .contiguous()[grid_batch, grid_x, grid_y]
        .permute(0, 3, 1, 2)
    )
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.shape[2] * ratio + 0.5), int(x.shape[3] * ratio + 0.5)
    offset_x = tf.random.uniform(
        [x.shape[0], 1, 1],
        minval=0,
        maxval=x.shape[2] + (1 - cutout_size[0] % 2),
        dtype=tf.dtypes.int32,
    )
    offset_y = tf.random.uniform(
        [x.shape[0], 1, 1],
        minval=0,
        maxval=x.shape[3] + (1 - cutout_size[1] % 2),
        dtype=tf.dtypes.int32,
    )
    grid_batch, grid_x, grid_y = tf.meshgrid(
        tf.range(x.shape[0], dtype=tf.dtypes.int64),
        tf.range(cutout_size[0], dtype=tf.dtypes.int64),
        tf.range(cutout_size[1], dtype=tf.dtypes.int64),
    )
    grid_x = tf.clip_by_value(
        grid_x + offset_x - cutout_size[0] // 2, 0, x.shape[2] - 1
    )
    grid_y = tf.clip_by_value(
        grid_y + offset_y - cutout_size[1] // 2, 0, x.shape[3] - 1
    )
    mask = tf.ones([x.shape[0], x.shape[2], x.shape[3]], dtype=x.dtype)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}
