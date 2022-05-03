import tensorflow as tf
import lpips_tf

ploss = lpips_tf.PerceptualLoss(model="net-lin", net="alex")


def get_perceptual_loss(input0, input1):
    return tf.math.reduce_sum(ploss(input0, input1))


def resize(image, size):
    return tf.image.resize(image, [size, size])


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


# @tf.function
# def discrimination_loss(logits_real, logits_fake):
#     real_loss = tf.minimum(0.0, -1 + logits_real)
#     real_loss = -1 * tf.reduce_mean(real_loss)

#     fake_loss = tf.minimum(0.0, -1 - logits_fake)
#     fake_loss = -1 * tf.reduce_mean(fake_loss)

#     return real_loss + fake_loss


@tf.function
def discrimination_loss(logits_real, logits_fake):
    real_loss = tf.math.reduce_mean(
        tf.nn.relu(tf.random.uniform(tf.shape(logits_real)) * 0.2 + 0.8 - logits_real)
    )

    fake_loss = tf.math.reduce_mean(
        tf.nn.relu(tf.random.uniform(tf.shape(logits_fake)) * 0.2 + 0.8 + logits_fake)
    )

    return real_loss + fake_loss


@tf.function
def reconstruction_loss(real_image, rec_image, rec_small, rec_part, part):
    return (
        get_perceptual_loss(rec_image, resize(real_image, tf.shape(rec_image)[1]),)
        + get_perceptual_loss(rec_small, resize(real_image, tf.shape(rec_small)[1]),)
        + get_perceptual_loss(
            rec_part,
            resize(crop_image_by_part(real_image, part), tf.shape(rec_part)[1]),
        )
    )


@tf.function
def generator_loss(logits_fake):
    return -1 * tf.reduce_mean(logits_fake)
