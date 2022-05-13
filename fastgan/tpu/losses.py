import tensorflow as tf


@tf.function
def prediction_loss(logits_real, logits_fake):
    real_loss = tf.nn.relu(1.0 + logits_real)

    fake_loss = tf.nn.relu(1.0 - logits_fake)

    return tf.math.reduce_mean(fake_loss + real_loss)


@tf.function
def reconstruction_loss(real_image, rec_image):
    size = tf.shape(rec_image)[1]
    return tf.keras.metrics.mean_squared_error(
        tf.image.resize(real_image, [size, size]), rec_image
    )


@tf.function
def generator_loss(logits_fake):
    return tf.reduce_mean(logits_fake)
