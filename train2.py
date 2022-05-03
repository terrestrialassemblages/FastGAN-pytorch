from tkinter import Image
from numpy import real
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.utils as kutils
from tensorflow.keras import optimizers

import argparse
import random
from tqdm import tqdm

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


def postprocess_images(images, dtype=tf.float32):
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
        seed=42,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.generator_save_path = generator_save_path
        self.manager = model_manager

        self.writer = tf.summary.create_file_writer(log_dir)

        self.real_images = real_images
        self.fixed_noise = tf.random.normal((8, nz), 0, 1, seed=seed)

    def on_epoch_end(self, epoch, logs=None):
        model_pred_fnoise = self.model.netG(self.fixed_noise)
        grid_gen = postprocess_images(imgrid(model_pred_fnoise[0], 8))

        part = random.randint(0, 3)
        _, rec_imgs = self.model.netD(self.real_images, "real", part=part)
        rec_imgs = tf.concat(
            [tf.image.resize(self.real_images, [128, 128]), *rec_imgs], axis=0,
        )
        grid_rec = postprocess_images(imgrid(rec_imgs, 8))

        kutils.save_img(saved_image_folder + f"/gen_{epoch}.jpg", grid_gen, scale=False)
        # kutils.save_img(saved_image_folder + f"/rec_{epoch}.jpg", grid_rec, scale=False)

        self.model.netG.save(generator_save_path)
        self.manager.save()

        self.manager.checkpoint.epoch.assign(epoch)
        with self.writer.as_default():
            tf.summary.scalar("GLoss/Generator", logs["gen_loss"], step=epoch)
            tf.summary.scalar(
                "DLoss/Reconstruction", logs["rec_loss"], step=epoch
            )
            tf.summary.scalar(
                "DLoss/Discriminator", logs["disc_loss"], step=epoch
            )
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

        self.gloss_tracker = keras.metrics.Mean(name="gen_loss")
        self.dloss_tracker = keras.metrics.Mean(name="disc_loss")
        self.rloss_tracker = keras.metrics.Mean(name="rec_loss")
        self.policy = data_policy

    @property
    def metrics(self):
        return [self.gloss_tracker, self.dloss_tracker, self.rloss_tracker]

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.optimizerD = d_optimizer
        self.optimizerG = g_optimizer

    @tf.function
    def train_step(self, real_images):
        current_batch_size = tf.shape(real_images)[0]

        with tf.GradientTape() as tapeG, tf.GradientTape() as tapeD:
            noise = tf.random.normal((current_batch_size, self.nz), 0, 1)
            fake_images = self.netG(noise, training=True)

            real_images = DiffAugment(real_images, policy=self.policy)
            fake_images = [
                DiffAugment(fake, policy=self.policy) for fake in fake_images
            ]

            part = random.randint(0, 3)
            d_logits_on_real, rec_imgs = self.netD(real_images, "real", part=part)
            d_logits_on_fake = self.netD(fake_images, "fake")

            disc_loss = losses.discrimination_loss(
                d_logits_on_real, d_logits_on_fake
            )
            rec_loss = losses.reconstruction_loss(
                real_images, *rec_imgs, part=part
            )

            lossD = disc_loss + rec_loss

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
        self.dloss_tracker.update_state(disc_loss)
        self.rloss_tracker.update_state(rec_loss)

        return {m.name: m.result() for m in self.metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="region gan")
    parser.add_argument(
        "--path",
        type=str,
        default="../apebase/ipfs",
        help="path of resource dataset, should be a folder that has one or many sub image folders inside",
    )
    parser.add_argument("--name", type=str, default="", help="experiment name")
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs")
    parser.add_argument(  # 8
        "--batch_size", type=int, default=8, help="mini batch number of images"
    )
    parser.add_argument(
        "--resume", action="store_true", help="whether to resume training"
    )
    parser.add_argument(
        "--data_aug_policy", type=str, default="color,translation", help=""
    )
    parser.add_argument("--im_size", type=int, default=256, help="image resolution")
    parser.add_argument("--ngf", type=int, default=16, help="")
    parser.add_argument("--ndf", type=int, default=16, help="")
    parser.add_argument("--nz", type=int, default=256, help="")
    parser.add_argument("--nlr", type=float, default=0.0002, help="")
    parser.add_argument("--nbeta1", type=float, default=0.5, help="")
    parser.add_argument("--save_interval", type=int, default=100, help="")
    parser.add_argument("--shuffle_buffer", type=int, default=256, help="")

    args = parser.parse_args()
    print(args)

    experiment_name = args.name or f"nf{args.ngf}_nz{args.nz}_imsize{args.im_size}_"
    saved_model_folder, saved_image_folder, log_folder = get_dir(args, experiment_name)
    generator_save_path = saved_model_folder + f"/generator"

    model = FastGan(
        args.ngf, args.ndf, args.nz, 3, args.im_size, data_policy=args.data_aug_policy
    )
    model.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=args.nlr, beta_1=args.nbeta1),
        g_optimizer=keras.optimizers.Adam(learning_rate=args.nlr, beta_1=args.nbeta1),
    )

    checkpoint = tf.train.Checkpoint(
        netG=model.netG,
        netD=model.netD,
        optimizerG=model.optimizerG,
        optimizerD=model.optimizerD,
        epoch=tf.Variable(0, name="epoch"),
    )

    manager = tf.train.CheckpointManager(checkpoint, saved_model_folder, max_to_keep=3)

    if manager.latest_checkpoint and args.resume:
        checkpoint.restore(manager.latest_checkpoint)

    ds = kutils.image_dataset_from_directory(
        "../apebase/ipfs",
        image_size=(args.im_size, args.im_size),
        labels=None,
        batch_size=None,
    )
    ds = ds.map(preprocess_images, num_parallel_calls=tf.data.AUTOTUNE).shuffle(
        args.shuffle_buffer
    )
    ds = ds.batch(args.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    callback = TrainingCallback(
        log_folder,
        saved_image_folder,
        manager,
        generator_save_path,
        next(iter(ds)),
        args.nz,
    )
    try:
        model.fit(
            ds, epochs=args.epochs, callbacks=[callback],
        )
    except KeyboardInterrupt:
        print("RECEIVED KEYBOARD INTERRUPT, SAVING MODEL")
        manager.save()
        model.netG.save(generator_save_path)
        exit(1)

