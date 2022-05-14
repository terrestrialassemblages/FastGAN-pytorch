import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.utils as kutils

from fastgan.tpu import losses
from fastgan.tpu.models import Discriminator, Generator
from fastgan.operation import imgrid, get_dir
from fastgan.diffaug import DiffAugment


class Args:
    def __init__(
        self,
        batch_size=128,
        bucket="apebase-training",
        data_aug_policy="color,cutout",
        data_path="apebase.tfrecords",
        ds_len=10000,
        epochs=1000,
        gp_weight=0,
        im_save_interval=1,
        im_size=256,
        lr=0.0002,
        name="",
        nbeta1=0.5,
        nc=3,
        ndf=64,
        ngf=64,
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
        self.gp_weight = gp_weight
        self.im_save_interval = im_save_interval
        self.im_size = im_size
        self.lr = lr
        self.name = name or f"nf{ngf}_nz{nz}_imsize{im_size}_"
        self.nbeta1 = nbeta1
        self.nc = nc
        self.ndf = ndf
        self.ngf = ngf
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


def preprocess_images(images, random_flip=True, dtype=tf.float32):
    if random_flip:
        images = tf.image.random_flip_left_right(images)
    images = tf.cast(images, dtype) - 127.5
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

            _, rec_img = self.model.netD(self.real_images, training=False)
            rec_img = tf.concat(
                [
                    tf.image.resize(self.real_images, [128, 128], method="nearest"),
                    rec_img,
                ],
                axis=0,
            )
            grid_rec = postprocess_images(imgrid(rec_img, 8))

            kutils.save_img(
                self.image_dir + f"/gen_{epoch:05}.jpg", grid_gen, scale=False
            )
            kutils.save_img(
                self.image_dir + f"/rec_{epoch:05}.jpg", grid_rec, scale=False
            )

            self.model.netG.save_weights(
                self.generator_save_path, options=self.save_options
            )
            self.manager.save(options=self.save_options)


class FastGan(keras.Model):
    def __init__(
        self,
        ngf,
        ndf,
        nz,
        nc,
        im_size,
        data_policy="",
        gp_weight=10,
        precision=tf.float32,
    ):
        super().__init__()
        self.nz = nz
        self.netG = Generator(ngf=ngf, nz=nz, nc=nc, im_size=im_size)
        self.netD = Discriminator(ndf=ndf, nc=nc, im_size=im_size)
        self.policy = data_policy

        self.gp_weight = gp_weight
        self.precision = precision

    @property
    def metrics(self):
        return [
            self.gloss_tracker,
            self.dloss_tracker,
            self.rloss_tracker,
        ]

    def compile(self, d_optimizer, g_optimizer, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.optimizerD = d_optimizer
        self.optimizerG = g_optimizer

        self.gloss_tracker = keras.metrics.Mean(name="gen_loss")
        self.dloss_tracker = keras.metrics.Mean(name="pred_loss")
        self.rloss_tracker = keras.metrics.Mean(name="rec_loss")

    # @tf.function
    def train_step(self, real_images):
        current_batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(
            (current_batch_size, self.nz), 0, 1, dtype=self.precision
        )

        with tf.GradientTape() as tapeG, tf.GradientTape() as tapeD:
            fake_images = self.netG(noise, training=True)

            real_images = DiffAugment(real_images, policy=self.policy)
            fake_images = DiffAugment(fake_images, policy=self.policy)

            d_logits_on_real, rec_img = self.netD(real_images, training=True)
            d_logits_on_fake, _ = self.netD(fake_images, training=True)

            pred_loss = losses.prediction_loss(d_logits_on_real, d_logits_on_fake)
            rec_loss = losses.reconstruction_loss(real_images, rec_img)

            lossD = pred_loss + rec_loss
            lossG = losses.generator_loss(d_logits_on_fake)

            if self.gp_weight > 0:
                alpha = tf.random.uniform(
                    [tf.shape(real_images)[0], 1, 1, 1], minval=0.0, maxval=1.0
                )
                interpolation = alpha * real_images + (1 - alpha) * fake_images
                with tf.GradientTape() as tapeP:
                    tapeP.watch(interpolation)
                    logits, _ = self.netD(interpolation, training=True)

                gradients = tapeP.gradient(logits, interpolation)
                norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
                gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2) * self.gp_weight
                lossD += gradient_penalty

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
        model = FastGan(
            args.ngf,
            args.ndf,
            args.nz,
            args.nc,
            args.im_size,
            data_policy=args.data_aug_policy,
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
