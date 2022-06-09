import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.utils as kutils

import argparse

import lwgan.losses as losses
from lwgan.models import Discriminator, Generator
from util.operation import get_dir, imgrid
from util.diffaug import DiffAugment
from util.gradient_accumulation import GradientAccumulator


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
        self.save_interval = 100

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

    def on_epoch_end(self, epoch, _):
        epoch = epoch + 1  # because end of epoch
        model_pred_fnoise = self.model.netG(self.fixed_noise, training=False)
        grid_gen = postprocess_images(imgrid(model_pred_fnoise, 8))

        kutils.save_img(self.image_dir + f"/gen_{epoch:05}.jpg", grid_gen, scale=False)

        self.model.netG.save_weights(self.generator_save_path)
        self.manager.save()

        self.epoch = epoch
        self.manager.checkpoint.epoch.assign(epoch)
        with self.writer.as_default():
            tf.summary.image("Generated images", [grid_gen], step=epoch)


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
        self.step = 0

    @property
    def metrics(self):
        return [
            self.gloss_tracker,
            self.ploss_tracker,
            self.rloss_tracker,
            self.gp_tracker,
        ]

    def compile(self, d_optimizer, g_optimizer, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.optimizerD = d_optimizer
        self.optimizerG = g_optimizer

        self.gloss_tracker = keras.metrics.Mean(name="gen_loss")
        self.ploss_tracker = keras.metrics.Mean(name="pred_loss")
        self.rloss_tracker = keras.metrics.Mean(name="rec_loss")
        self.gp_tracker = keras.metrics.Mean(name="gp")

    @tf.function
    def train_step(self, real_images_batch):
        for i in range(self.n_gradient_accumulation):
            real_images = real_images_batch[i]
            current_batch_size = tf.shape(real_images)[0]

            noise = tf.random.normal((current_batch_size, self.nz), 0, 1)
            fake_images = self.netG(noise)

            with tf.GradientTape() as tapeD:
                d_logits_on_fake, d_logits_on_fake_32, _ = self.netD_aug(fake_images)
                d_logits_on_real, d_logits_on_real_32, rec_loss = self.netD_aug(
                    real_images, calc_aux_loss=True
                )

                pred_loss = losses.prediction_loss(
                    d_logits_on_real, d_logits_on_fake
                ) + losses.prediction_loss(d_logits_on_real_32, d_logits_on_fake_32)

                lossD = pred_loss + rec_loss
                lossD /= self.n_gradient_accumulation
                
                self.ploss_tracker.update_state(pred_loss / self.n_gradient_accumulation)
                self.rloss_tracker.update_state(rec_loss / self.n_gradient_accumulation)

                # if self.gp_weight > 0 and self.step % 4 == 0:
                #     alpha = tf.random.normal([current_batch_size, 1, 1, 1])
                #     diff = fake_images - real_images
                #     interpolated = real_images + alpha * diff

                #     with tf.GradientTape() as tapeGP:
                #         tapeGP.watch(interpolated)
                #         pred, pred_32, _ = self.netD_aug(interpolated)

                #     grads = tapeGP.gradient([pred, pred_32], [interpolated])[0]
                #     norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
                #     gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)

                #     self.gp_tracker.update_state(gradient_penalty)
                #     lossD = lossD + gradient_penalty

            self.optimizerD.apply_gradients(
                zip(
                    tapeD.gradient(lossD, self.netD.trainable_variables),
                    self.netD.trainable_variables,
                )
            )

        for i in range(self.n_gradient_accumulation):
            noise = tf.random.normal((current_batch_size, self.nz), 0, 1)
            with tf.GradientTape() as tapeG:
                fake_images = self.netG(noise, training=True)
                d_logits_on_fake, d_logits_on_fake_32, _ = self.netD_aug(fake_images)

                lossG = losses.generator_loss(d_logits_on_fake) + losses.generator_loss(
                    d_logits_on_fake_32
                )
                lossG /= self.n_gradient_accumulation
                self.gloss_tracker.update_state(lossG)

            self.optimizerG.apply_gradients(
                zip(
                    tapeG.gradient(lossG, self.netG.trainable_variables),
                    self.netG.trainable_variables,
                )
            )

        self.step += 1

        return {m.name: m.result() for m in self.metrics}


if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)
    parser = argparse.ArgumentParser(description="region gan")
    parser.add_argument(
        "--path",
        type=str,
        default="apebase.tfrecords",
        help="path of resource dataset, should be a folder that has one or many sub image folders inside",
    )
    parser.add_argument(
        "--ds_len", type=int, default=10000, help="number of training examples"
    )
    parser.add_argument("--name", type=str, default="", help="experiment name")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="mini batch number of images"
    )
    parser.add_argument(
        "--resume", action="store_true", help="whether to resume training"
    )
    parser.add_argument("--data_aug_policy", type=str, default="color,cutout", help="")
    parser.add_argument("--data_aug_prob", type=float, default=0.35, help="")
    parser.add_argument("--im_size", type=int, default=128, help="image resolution")
    parser.add_argument("--fmap_max", type=int, default=128, help="")
    parser.add_argument("--nz", type=int, default=256, help="")
    parser.add_argument("--lr", type=float, default=0.0002, help="")
    parser.add_argument("--nbeta1", type=float, default=0.5, help="")
    parser.add_argument("--shuffle_buffer", type=int, default=256, help="")
    parser.add_argument("--random_flip", type=bool, default=False, help="")
    parser.add_argument("--n_gradient_accumulation", type=int, default=4, help="")
    parser.add_argument("--gp_weight", type=int, default=10, help="")

    args = parser.parse_args()
    print(args)

    experiment_name = args.name or f"fmax{args.fmap_max}_nz{args.nz}_imsize{args.im_size}"
    saved_model_folder, saved_image_folder, log_folder = get_dir(args, experiment_name)
    generator_save_path = saved_model_folder + f"/generator.h5"

    model = LightweightGan(
        args.fmap_max,
        args.nz,
        3,
        args.im_size,
        aug_policy=args.data_aug_policy,
        aug_prob=args.data_aug_prob,
        n_gradient_accumulation=args.n_gradient_accumulation,
        gp_weight=args.gp_weight,
    )
    model.compile(
        d_optimizer=GradientAccumulator(
            keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.nbeta1),
            accum_steps=args.n_gradient_accumulation,
        ),
        g_optimizer=GradientAccumulator(
            keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.nbeta1),
            accum_steps=args.n_gradient_accumulation,
        ),
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

    def process_ds(record_bytes, dtype=tf.float32):
        image = tf.io.parse_single_example(
            record_bytes, {"image_raw": tf.io.FixedLenFeature([], tf.string)}
        )["image_raw"]
        image = tf.io.decode_png(image, channels=3)
        image = tf.image.resize(image, [args.im_size, args.im_size])
        if args.random_flip:
            image = tf.image.random_flip_left_right(image)
        image = tf.cast(image, dtype) - 127.5
        image = image / 127.5
        return image

    ds = tf.data.TFRecordDataset(args.path)
    ds = (
        ds.map(process_ds, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(args.shuffle_buffer)
        .repeat()
    )
    ds = (
        ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .batch(args.n_gradient_accumulation)
    )

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
            steps_per_epoch=args.ds_len
            // args.batch_size
            // args.n_gradient_accumulation,
            callbacks=[training_callback],
        )
    except KeyboardInterrupt:
        print("RECEIVED KEYBOARD INTERRUPT, SAVING MODEL")
        manager.save()
        model.netG.save_weights(generator_save_path)
        exit(1)

