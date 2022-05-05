import tensorflow as tf
from models import LpipsNetwork

lpips = LpipsNetwork()


def resize_a_to_b(a, b):
    size = tf.shape(b)[1]
    return tf.image.resize(a, [size, size])


def crop_image_by_part(image, part):
    hw = tf.shape(image)[1] // 2
    if part == 0:
        return image[:, :hw, :hw, :]
    if part == 1:
        return image[:, :hw, hw:, :]
    if part == 2:
        return image[:, hw:, :hw, :]
    if part == 3:
        return image[:, hw:, hw:, :]


@tf.function
def prediction_loss(logits_real, logits_fake):
    real_loss = tf.math.reduce_mean(tf.nn.relu(1.0 - logits_real))

    fake_loss = tf.math.reduce_mean(tf.nn.relu(1.0 + logits_fake))

    return fake_loss + real_loss


@tf.function
def reconstruction_loss(real_image, rec_image, rec_part, part):
    return lpips(resize_a_to_b(real_image, rec_image), rec_image,) + lpips(
        resize_a_to_b(crop_image_by_part(real_image, part), rec_part), rec_part,
    )


@tf.function
def generator_loss(logits_fake):
    return -tf.reduce_mean(logits_fake)
