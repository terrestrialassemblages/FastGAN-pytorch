import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.utils as kutils
from tqdm import tqdm

import random

import losses
from models import Discriminator, Generator
from operation import imgrid, get_dir
from diffaug import DiffAugment


class Args:
    def __init__(
        self,
        data_path="apebase.tfrecords",
        bucket="apebase-training",
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
        im_save_interval=100,
        seed=42,
    ):
        self.data_path = data_path
        self.bucket = bucket
        self.ds_len = ds_len
        self.name = name or f"nf{ngf}_nz{nz}_imsize{im_size}_"
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
        self.im_save_interval = im_save_interval
        self.seed = seed

        self.steps_per_epoch = steps_per_epoch or (ds_len // batch_size)
        self.steps_per_execution = steps_per_execution or steps_per_epoch // 2


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


# def get_dir(args: Args):
#     prefix = f"train_results/{args.name}/"
#     client = gcs.Client()

#     bucket = client.get_bucket(args.bucket)
#     for blob in bucket.list_blobs(prefix=prefix):
#         blob.delete()

#     full_path = f"gs://{args.bucket}/{prefix}"

#     saved_model_folder = full_path + "models/"
#     saved_image_folder = full_path + "images/"
#     log_folder = full_path + "logs/"

#     return saved_model_folder, saved_image_folder, log_folder


class TrainingCallback(keras.callbacks.Callback):
    def __init__(
        self, args, log_dir, image_dir, generator_save_path, model_manager, real_images,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.generator_save_path = generator_save_path
        self.manager = model_manager

        self.writer = tf.summary.create_file_writer(log_dir)

        self.real_images = real_images
        self.fixed_noise = tf.random.normal((8, args.nz), 0, 1, seed=args.seed)

        self.dataset_cardinality = args.ds_len
        self.im_save_interval = args.im_save_interval

    def on_train_batch_end(self, batch, logs=None):
        step = batch + self.dataset_cardinality * self.manager.checkpoint.epoch.numpy()
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

    def on_epoch_end(self, _):
        self.manager.checkpoint.epoch.assign_add(1)
        epoch = self.manager.checkpoint.epoch.numpy()
        if epoch % self.im_save_interval == 0:
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

            with self.writer.as_default():
                tf.summary.image("Generated images", [grid_gen], step=epoch)
                tf.summary.image(
                    "Reconstructed images", [grid_rec], step=epoch,
                )


class FastGan(keras.Model):
    def __init__(
        self,
        ngf,
        ndf,
        nz,
        nc,
        im_size,
        data_policy="color,translation",
        rec_weight=1,
        pen_weight=0.001,
    ):
        super().__init__()
        self.nz = nz
        self.netG = Generator(ngf=ngf, nz=nz, nc=nc, im_size=im_size)
        self.netD = Discriminator(ndf=ndf, nc=nc, im_size=im_size)
        self.policy = data_policy

        self.rec_weight = rec_weight
        self.pen_weight = pen_weight

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

    @tf.function
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

            part = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
            d_logits_on_real, rec_imgs = self.netD(real_aug, part, training=True)
            d_logits_on_fake, _ = self.netD(fake_aug, part, training=True)

            pred_loss = losses.prediction_loss(d_logits_on_real, d_logits_on_fake)
            rec_loss = (
                losses.reconstruction_loss(real_aug, *rec_imgs, part=part)
                * self.rec_weight
            )
            grad_penalty = (
                self.gradient_penalty(real_images, fake_images) * self.pen_weight
            )

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


def main(args: Args, resolver: tf.distribute.cluster_resolver.TPUClusterResolver):
    saved_model_folder, saved_image_folder, log_folder = get_dir(args, args.name)
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

    ds = tf.data.TFRecordDataset(f"gs://{args.bucket}/{args.data_path}")
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
        args,
        log_folder,
        saved_image_folder,
        generator_save_path,
        manager,
        next(iter(ds)),
    )

    try:
        model.fit(
            ds,
            epochs=total_epochs,
            steps_per_epoch=args.steps_per_epoch,
            callbacks=[training_callback],
        )
    except Exception as e:
        print("RECEIVED INTERRUPT, SAVING MODEL")
        manager.save()
        model.netG.save_weights(generator_save_path)
        raise e

