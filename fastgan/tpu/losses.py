import tensorflow as tf
from .models import LpipsNetwork

lpips = LpipsNetwork()


def resize_a_to_b(a, b):
    size = tf.shape(b)[1]
    return tf.image.resize(a, [size, size])


@tf.function
def prediction_loss(logits_real, logits_fake):
    real_loss = tf.math.reduce_mean(tf.nn.relu(1.0 - logits_real))

    fake_loss = tf.math.reduce_mean(tf.nn.relu(1.0 + logits_fake))

    return fake_loss + real_loss


@tf.function
def reconstruction_loss(real_image, rec_image):
    return lpips(resize_a_to_b(real_image, rec_image), rec_image)


@tf.function
def generator_loss(logits_fake):
    return -tf.reduce_mean(logits_fake)
