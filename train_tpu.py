import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.utils as kutils

import random

import losses
from models import Discriminator, Generator
from operation import get_dir, imgrid
from diffaug import DiffAugment


def preprocess_images(images, random_flip=True):
    if random_flip:
        images = tf.image.random_flip_left_right(images)
    images = tf.cast(images, tf.float32) - 127.5
    images = images / 127.5
    return images


def postprocess_images(images, dtype=tf.uint8):
    images = (images * 127.5) + 127.5
    images = tf.cast(images, dtype)
    return images


class TrainingCallback(keras.callbacks.Callback):
    def __init__(
        self,
        log_dir,
        image_dir,
        model_manager,
        generator_save_path,
        real_images,
        nz,
        dataset_cardinality,
        seed=42,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.generator_save_path = generator_save_path
        self.manager = model_manager

        self.writer = tf.summary.create_file_writer(log_dir)

        self.real_images = real_images
        self.fixed_noise = tf.random.normal((8, nz), 0, 1, seed=seed)

        self.dataset_cardinality = dataset_cardinality
        self.epoch = 0

    def on_train_batch_end(self, batch, logs=None):
        step = batch + self.dataset_cardinality * self.epoch
        with self.writer.as_default():
            tf.summary.scalar("Generator/Loss", logs["gen_loss"], step=step)
            tf.summary.scalar(
                "Discriminator/Loss", logs["pred_loss"] + logs["rec_loss"], step=step
            )
            tf.summary.scalar(
                "Discriminator/Reconstruction Loss", logs["rec_loss"], step=step
            )
            tf.summary.scalar(
                "Discriminator/Prediction Loss", logs["pred_loss"], step=step
            )
            tf.summary.scalar(
                "Discriminator/Grad Penalty", logs["grad_penalty"], step=step
            )

    def on_epoch_end(self, epoch, _):
        epoch = epoch + 1  # because end of epoch
        model_pred_fnoise = self.model.netG(self.fixed_noise, training=False)
        grid_gen = postprocess_images(imgrid(model_pred_fnoise, 8))

        part = random.randint(0, 3)
        _, rec_imgs = self.model.netD(self.real_images, part=part, training=False)
        rec_imgs = tf.concat(
            [tf.image.resize(self.real_images, [128, 128]), *rec_imgs], axis=0,
        )
        grid_rec = postprocess_images(imgrid(rec_imgs, 8))

        kutils.save_img(self.image_dir + f"/gen_{epoch}.jpg", grid_gen, scale=False)
        kutils.save_img(self.image_dir + f"/rec_{epoch}.jpg", grid_rec, scale=False)

        self.model.netG.save_weights(self.generator_save_path)
        self.manager.save()

        self.epoch = epoch
        self.manager.checkpoint.epoch.assign(epoch)
        with self.writer.as_default():
            tf.summary.image("Generated images", [grid_gen], step=epoch)
            tf.summary.image(
                "Reconstructed images", [grid_rec], step=epoch,
            )


class FastGan(keras.Model):
    def __init__(self, ngf, ndf, nz, nc, im_size, data_policy="color,translation"):
        super().__init__()
        self.nz = nz
        self.netG = Generator(ngf=ngf, nz=nz, nc=nc, im_size=im_size)
        self.netD = Discriminator(ndf=ndf, nc=nc, im_size=im_size)
        self.policy = data_policy

    @property
    def metrics(self):
        return [
            self.gloss_tracker,
            self.dloss_tracker,
            self.rloss_tracker,
            self.gp_tracker,
        ]

    def compile(self, d_optimizer, g_optimizer, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.optimizerD = d_optimizer
        self.optimizerG = g_optimizer

        self.gloss_tracker = keras.metrics.Mean(name="gen_loss")
        self.dloss_tracker = keras.metrics.Mean(name="pred_loss")
        self.rloss_tracker = keras.metrics.Mean(name="rec_loss")
        self.gp_tracker = keras.metrics.Mean(name="grad_penalty")

    def gradient_penalty(self, real_samples, fake_samples):
        alpha = tf.random.uniform(
            [tf.shape(real_samples)[0], 1, 1, 1], minval=0.0, maxval=1.0
        )
        diff = fake_samples - real_samples
        interpolation = real_samples + alpha * diff

        with tf.GradientTape() as gradient_tape:
            gradient_tape.watch(interpolation)
            logits = self.netD(
                DiffAugment(interpolation, policy=self.policy), training=True
            )

        gradients = gradient_tape.gradient(logits, [interpolation])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
        return gradient_penalty

    @tf.function
    def train_step(self, real_images):
        current_batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal((current_batch_size, self.nz), 0, 1)
        with tf.GradientTape() as tapeG, tf.GradientTape() as tapeD:
            fake_images = self.netG(noise, training=True)

            real_aug = DiffAugment(real_images, policy=self.policy)
            fake_aug = DiffAugment(fake_images, policy=self.policy)

            part = random.randint(0, 3)
            d_logits_on_real, rec_imgs = self.netD(real_aug, part=part, training=True)
            d_logits_on_fake = self.netD(fake_aug, training=True)

            pred_loss = losses.prediction_loss(d_logits_on_real, d_logits_on_fake)
            rec_loss = (
                losses.reconstruction_loss(real_aug, *rec_imgs, part=part) * 0.001
            )
            grad_penalty = self.gradient_penalty(real_images, fake_images) * 0.001

            lossD = pred_loss + rec_loss + grad_penalty

            lossG = losses.generator_loss(d_logits_on_fake)

        self.optimizerD.apply_gradients(
            zip(
                tapeD.gradient(lossD, self.netD.trainable_variables),
                self.netD.trainable_variables,
            )
        )
        self.optimizerG.apply_gradients(
            zip(
                tapeG.gradient(lossG, self.netG.trainable_variables),
                self.netG.trainable_variables,
            )
        )

        # Monitor loss.
        self.gloss_tracker.update_state(lossG)
        self.dloss_tracker.update_state(pred_loss)
        self.rloss_tracker.update_state(rec_loss)
        self.gp_tracker.update_state(grad_penalty)

        return {m.name: m.result() for m in self.metrics}


class Args:
    def __init__(
        self,
        path="apebase.tfrecords",
        ds_len=10000,
        name="",
        epochs=10000,
        batch_size=1024,
        resume=False,
        data_aug_policy="color,cutout",
        im_size=256,
        ngf=64,
        ndf=64,
        nz=256,
        lr=0.0002,
        nbeta1=0.5,
        random_flip=False,
        steps_per_epoch=None,
        steps_per_execution=3,
    ):
        self.path = path
        self.ds_len = ds_len
        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size
        self.resume = resume
        self.data_aug_policy = data_aug_policy
        self.im_size = im_size
        self.ngf = ngf
        self.ndf = ndf
        self.nz = nz
        self.lr = lr
        self.nbeta1 = nbeta1
        self.random_flip = random_flip

        self.steps_per_epoch = steps_per_epoch or (self.ds_len // self.batch_size)
        self.steps_per_execution = steps_per_execution or self.steps_per_epoch // 2


def main(args: Args, resolver: tf.distribute.cluster_resolver.TPUClusterResolver):
    experiment_name = args.name or f"nf{args.ngf}_nz{args.nz}_imsize{args.im_size}_"
    saved_model_folder, saved_image_folder, log_folder = get_dir(args, experiment_name)
    generator_save_path = saved_model_folder + f"/generator.h5"

    def process_ds(record_bytes):
        image = tf.io.parse_single_example(
            record_bytes, {"image_raw": tf.io.FixedLenFeature([], tf.string)}
        )["image_raw"]
        image = tf.io.decode_png(image, channels=3)
        image = tf.image.resize(image, [args.im_size, args.im_size], method="nearest")
        if args.random_flip:
            image = tf.image.random_flip_left_right(image)
        image = tf.cast(image, tf.float32) - 127.5
        image = image / 127.5
        return image

    ds = tf.data.TFRecordDataset(args.path)
    ds = ds.map(process_ds, num_parallel_calls=tf.data.AUTOTUNE).shuffle(args.ds_len)
    ds = ds.repeat().prefetch(buffer_size=tf.data.AUTOTUNE).batch(args.batch_size)

    strategy = tf.distribute.TPUStrategy(resolver)

    with strategy.scope():
        model = FastGan(
            args.ngf,
            args.ndf,
            args.nz,
            3,
            args.im_size,
            data_policy=args.data_aug_policy,
        )
        model.compile(
            d_optimizer=keras.optimizers.Adam(
                learning_rate=args.lr, beta_1=args.nbeta1
            ),
            g_optimizer=keras.optimizers.Adam(
                learning_rate=args.lr, beta_1=args.nbeta1
            ),
            steps_per_execution=args.steps_per_execution,
        )

    checkpoint = tf.train.Checkpoint(
        netG=model.netG,
        netD=model.netD,
        optimizerG=model.optimizerG,
        optimizerD=model.optimizerD,
        epoch=tf.Variable(0, name="epoch"),
    )

    manager = tf.train.CheckpointManager(checkpoint, saved_model_folder, max_to_keep=3)

    total_epochs = args.epochs
    if manager.latest_checkpoint and args.resume:
        checkpoint.restore(manager.latest_checkpoint)
        total_epochs = total_epochs - checkpoint.epoch.numpy()

    training_callback = TrainingCallback(
        log_folder,
        saved_image_folder,
        manager,
        generator_save_path,
        next(iter(ds)),
        args.nz,
        args.ds_len,
    )

    try:
        model.fit(
            ds,
            epochs=total_epochs,
            steps_per_epoch=args.steps_per_epoch,
            callbacks=[training_callback],
        )
    except:
        print("RECEIVED INTERRUPT, SAVING MODEL")
        manager.save()
        model.netG.save_weights(generator_save_path)

