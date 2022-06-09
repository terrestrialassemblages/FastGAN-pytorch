import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.utils as kutils

import argparse

import lwgan.losses as losses
from lwgan.models import Discriminator, Generator
from util.operation import get_dir, imgrid
from util.diffaug import DiffAugment


class Args:
    def __init__(
        self,
        batch_size=128,
        bucket="apebase-training",
        data_aug_policy="color,cutout",
        data_path="apebase.tfrecords",
        ds_len=10000,
        epochs=1000,
        fmap_max=128,
        gp_weight=0,
        im_save_interval=1,
        im_size=256,
        lr=0.0002,
        n_gradient_accumulation=4,
        name="",
        nbeta1=0.5,
        nc=3,
        nz=256,
        precision="mixed_bfloat16",
        random_flip=False,
        resume=False,
        seed=42,
        steps_per_epoch=None,
        steps_per_execution=None,
    ):
        self.batch_size = batch_size
        self.bucket = bucket
        self.data_aug_policy = data_aug_policy
        self.data_path = data_path
        self.ds_len = ds_len
        self.epochs = epochs
        self.fmap_max = fmap_max
        self.gp_weight = gp_weight
        self.im_save_interval = im_save_interval
        self.im_size = im_size
        self.lr = lr
        self.n_gradient_accumulation = n_gradient_accumulation
        self.name = name or f"fmax{fmap_max}_nz{nz}_imsize{im_size}_"
        self.nbeta1 = nbeta1
        self.nc = nc
        self.nz = nz
        self.precision = precision
        self.random_flip = random_flip
        self.resume = resume
        self.seed = seed

        self.steps_per_epoch = steps_per_epoch or (ds_len // batch_size)
        self.steps_per_execution = steps_per_execution or self.steps_per_epoch // 6

    def get_precision_dtype(self):
        if self.precision == "mixed_bfloat16":
            return tf.bfloat16
        elif self.precision == "mixed_float16":
            return tf.float16
        else:
            return tf.float32


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
        self, args: Args, image_dir, saved_model_folder, model_manager, real_images,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.generator_save_path = saved_model_folder + "/generator.h5"
        self.manager = model_manager

        self.real_images = real_images
        self.fixed_noise = tf.random.normal(
            (32, args.nz), 0, 1, seed=args.seed, dtype=args.get_precision_dtype()
        )

        self.dataset_cardinality = args.ds_len
        self.im_save_interval = args.im_save_interval

        self.save_options = tf.saved_model.SaveOptions(
            experimental_io_device="/job:localhost"
        )

    def on_epoch_end(self, *_, **__):
        self.manager.checkpoint.epoch.assign_add(1)
        epoch = self.manager.checkpoint.epoch.numpy()
        if epoch % self.im_save_interval == 0:
            model_pred_fnoise = self.model.netG(self.fixed_noise, training=False)
            grid_gen = postprocess_images(imgrid(model_pred_fnoise, 8))

            kutils.save_img(
                self.image_dir + f"/gen_{epoch:05}.jpg", grid_gen, scale=False
            )

            self.model.netG.save_weights(
                self.generator_save_path, options=self.save_options
            )
            self.manager.save(options=self.save_options)


class LightweightGan(keras.Model):
    def __init__(
        self,
        fmap_max,
        nz,
        nc,
        im_size,
        aug_policy="color,translation",
        aug_prob=None,
        n_gradient_accumulation=4,
        gp_weight=10,
        precision=tf.float32,
    ):
        super().__init__()
        self.nz = nz
        self.netG = Generator(im_size, nchannels=nc, latent_dim=nz, fmap_max=fmap_max)
        self.netD = Discriminator(
            im_size, nchannels=nc, fmap_max=fmap_max, disc_output_size=1
        )
        self.netG.build((1, nz))
        self.netD.build((1, im_size, im_size, nc))

        self.netD_aug = lambda input_images, calc_aux_loss=False: self.netD(
            DiffAugment(input_images, policy=self.aug_policy, prob=self.aug_prob),
            calc_aux_loss=calc_aux_loss,
            training=True,
        )

        self.gradD = [
            tf.Variable(tf.zeros_like(x)) for x in self.netD.trainable_variables
        ]
        self.gradG = [
            tf.Variable(tf.zeros_like(x)) for x in self.netG.trainable_variables
        ]

        self.aug_policy = aug_policy
        self.aug_prob = aug_prob
        self.n_gradient_accumulation = n_gradient_accumulation
        self.gp_weight = gp_weight
        self.precision = precision
        self.step = 0

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
        self.gp_tracker = keras.metrics.Mean(name="gp")

    @tf.function
    def train_step(self, real_image_batch):
        for i in range(self.n_gradient_accumulation):
            real_images = real_image_batch[i]
            current_batch_size = tf.shape(real_images)[0]

            noise = tf.random.normal(
                (current_batch_size, self.nz), 0, 1, dtype=self.precision
            )
            fake_images = self.netG(noise)

            with tf.GradientTape() as tapeD:
                d_logits_on_fake, d_logits_on_fake_32, _ = self.netD_aug(fake_images)

                with tf.GradientTape() as tapeGP:
                    tapeGP.watch(real_images)
                    d_logits_on_real, d_logits_on_real_32, rec_loss = self.netD_aug(
                        real_images, calc_aux_loss=True
                    )
                    if self.gp_weight > 0 and self.step % 4 == 0:
                        outputs = [d_logits_on_real, d_logits_on_real_32]
                        gradients = tapeGP.gradient(
                            outputs,
                            real_images,
                            output_gradients=[tf.ones_like(x) for x in outputs],
                        )
                        gradients = tf.reshape(gradients, (current_batch_size, -1))
                        gradient_penalty = self.gp_weight * tf.reduce_mean(
                            (tf.norm(gradients, ord=2, axis=1) - 1) ** 2
                        )
                        self.gp_tracker.update_state(gradient_penalty)
                    else:
                        gradient_penalty = 0

                pred_loss = losses.prediction_loss(
                    d_logits_on_real, d_logits_on_fake
                ) + losses.prediction_loss(d_logits_on_real_32, d_logits_on_fake_32)

                lossD = pred_loss + rec_loss + gradient_penalty

                lossD /= self.n_gradient_accumulation
            gradients = tapeD.gradient(lossD, self.netD.trainable_variables)
            self.gradD = [
                acum_grad.assign_add(grad)
                for acum_grad, grad in zip(self.gradD, gradients)
            ]

        self.optimizerD.apply_gradients(zip(self.gradD, self.netD.trainable_variables))

        for _ in range(self.n_gradient_accumulation):
            noise = tf.random.normal(
                (current_batch_size, self.nz), 0, 1, dtype=self.precision
            )
            with tf.GradientTape() as tapeG:
                fake_images = self.netG(noise, training=True)
                d_logits_on_fake, d_logits_on_fake_32, _ = self.netD_aug(fake_images)

                lossG = losses.generator_loss(d_logits_on_fake) + losses.generator_loss(
                    d_logits_on_fake_32
                )

                lossG /= self.n_gradient_accumulation
            gradients = tapeG.gradient(lossG, self.netG.trainable_variables)
            self.gradG = [
                acum_grad.assign_add(grad)
                for acum_grad, grad in zip(self.gradG, gradients)
            ]

        self.optimizerG.apply_gradients(zip(self.gradG, self.netG.trainable_variables))

        for acum, var in zip(self.gradD, self.netD.trainable_variables):
            acum.assign(tf.zeros_like(var))
        for acum, var in zip(self.gradG, self.netG.trainable_variables):
            acum.assign(tf.zeros_like(var))

        self.step += 1

        # Monitor loss.
        self.gloss_tracker.update_state(lossG)
        self.dloss_tracker.update_state(pred_loss)
        self.rloss_tracker.update_state(rec_loss)

        return {m.name: m.result() for m in self.metrics}


def main(args: Args, resolver: tf.distribute.cluster_resolver.TPUClusterResolver):
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy(args.precision)

    saved_model_folder, saved_image_folder, _ = get_dir(args, args.name)

    def process_ds(record_bytes):
        image = tf.io.parse_single_example(
            record_bytes, {"image_raw": tf.io.FixedLenFeature([], tf.string)}
        )["image_raw"]
        image = tf.io.decode_png(image, channels=args.nc)
        image = tf.image.resize(image, [args.im_size, args.im_size], method="nearest")
        return preprocess_images(image, args.random_flip, args.get_precision_dtype())

    ds = tf.data.TFRecordDataset(f"gs://{args.bucket}/{args.data_path}")
    ds = ds.map(process_ds, num_parallel_calls=tf.data.AUTOTUNE).shuffle(args.ds_len)
    ds = ds.repeat().prefetch(buffer_size=tf.data.AUTOTUNE).batch(args.batch_size)

    strategy = tf.distribute.TPUStrategy(resolver)

    with strategy.scope():
        model = LightweightGan(
            args.fmap_max,
            args.nz,
            3,
            args.im_size,
            aug_policy=args.data_aug_policy,
            aug_prob=args.data_aug_prob,
            n_gradient_accumulation=args.n_gradient_accumulation,
            gp_weight=args.gp_weight,
            precision=args.get_precision_dtype(),
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
        saved_image_folder,
        saved_model_folder,
        manager,
        next(iter(ds.unbatch().batch(32))),
    )

    model.fit(
        ds,
        epochs=total_epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=[training_callback],
    )
