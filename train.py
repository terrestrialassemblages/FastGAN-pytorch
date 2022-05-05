import tensorflow as tf
import tensorflow.keras.utils as kutils
from tensorflow.keras import optimizers
from tensorflow.keras import layers

import argparse
import random
from tqdm import trange

import losses
from models import Discriminator, Generator
from operation import get_dir, imgrid
from diffaug import DiffAugment


def preprocess_images(images):
    images = tf.cast(images, tf.float32) - 127.5
    images = images / 127.5
    return images


def postprocess_images(images, dtype=tf.uint8):
    images = (images * 127.5) + 127.5
    images = tf.clip_by_value(images, 0, 255)
    return tf.cast(images, dtype)


@tf.function
def train_step(nz, real_images, netG, netD, optimizerG, optimizerD, policy):
    current_batch_size = tf.shape(real_images)[0]

    with tf.GradientTape() as tapeD, tf.GradientTape() as tapeG:
        noise = tf.random.normal((current_batch_size, nz), 0, 1)
        fake_images = netG(noise, training=True)

        real_images = DiffAugment(real_images, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]

        part = random.randint(0, 3)
        d_logits_on_real, rec_imgs = netD(real_images, "real", part=part)
        d_logits_on_fake = netD(fake_images, "fake")

        discrimination_loss = losses.prediction_loss(
            d_logits_on_real, d_logits_on_fake
        )
        reconstruction_loss = losses.reconstruction_loss(
            real_images, *rec_imgs, part=part
        )

        lossD = discrimination_loss + reconstruction_loss

        lossG = losses.generator_loss(d_logits_on_fake)

    optimizerD.apply_gradients(
        zip(tapeD.gradient(lossD, netD.trainable_variables), netD.trainable_variables,)
    )
    optimizerG.apply_gradients(
        zip(tapeG.gradient(lossG, netG.trainable_variables), netG.trainable_variables,)
    )

    return discrimination_loss, reconstruction_loss, lossD, lossG


def train(args):
    data_root = args.path
    total_iterations = args.iter
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = args.ndf  # 64
    ngf = args.ngf  # 64
    nz = args.nz  # 256
    nlr = args.nlr
    nbeta1 = args.nbeta1
    resume_training = args.resume
    save_interval = args.save_interval
    policy = args.data_aug_policy
    shuffle_buffer = args.shuffle_buffer
    experiment_name = args.name or f"nf{ngf}_nz{nz}_imsize{im_size}"

    saved_model_folder, saved_image_folder, log_folder = get_dir(args, experiment_name)
    summary_writer = tf.summary.create_file_writer(log_folder)

    ## load dataset and apply transforms
    ds = kutils.image_dataset_from_directory(
        data_root, image_size=(im_size, im_size), labels=None, batch_size=None
    )
    ds = (
        ds.map(preprocess_images, num_parallel_calls=tf.data.AUTOTUNE)
        .repeat()
        .shuffle(shuffle_buffer)
    )
    # ds = ds.apply(tf.data.experimental.copy_to_device("/gpu:0"))
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    dataloader = iter(ds)

    ## load networks and checkpoints
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netD = Discriminator(ndf=ndf, im_size=im_size)

    # forward passes to build model
    netG.initialize(batch_size)
    netD.initialize(batch_size)

    fixed_noise = tf.random.normal((8, nz), 0, 1, seed=42)

    optimizerG = optimizers.Adam(learning_rate=nlr, beta_1=nbeta1, beta_2=0.999)
    optimizerD = optimizers.Adam(learning_rate=nlr, beta_1=nbeta1, beta_2=0.999)

    checkpoint = tf.train.Checkpoint(
        netG=netG,
        netD=netD,
        optimizerG=optimizerG,
        optimizerD=optimizerD,
        step=tf.Variable(0, name="step"),
    )
    manager = tf.train.CheckpointManager(checkpoint, saved_model_folder, max_to_keep=3)
    if manager.latest_checkpoint and resume_training:
        checkpoint.restore(manager.latest_checkpoint)
        current_iteration = checkpoint.step.numpy()
        print(
            f"Load ckpt from {manager.latest_checkpoint} at step {checkpoint.step.numpy()}."
        )
    else:
        print("Training from scratch.")
        current_iteration = 0

    avg_param_G = netG.get_weights()

    # gracefully exit on ctrl+c
    try:
        for iteration in (
            prog_bar := trange(
                current_iteration,
                total_iterations + 1,
                initial=current_iteration,
                total=total_iterations + 1,
            )
        ) :
            real_images = next(dataloader)
            discrimination_loss, reconstruction_loss, lossD, lossG = train_step(
                nz, real_images, netG, netD, optimizerG, optimizerD, policy
            )

            for i, (w, avg_w) in enumerate(zip(netG.get_weights(), avg_param_G)):
                avg_param_G[i] = (avg_w * 0.999) + (0.001 * w)

            with summary_writer.as_default():
                tf.summary.scalar(
                    "DLoss/Reconstruction", discrimination_loss, step=iteration
                )
                tf.summary.scalar(
                    "DLoss/Discrimination", reconstruction_loss, step=iteration
                )
                tf.summary.scalar("DLoss/Total", lossD, step=iteration)
                tf.summary.scalar("GLoss/Generator", lossG, step=iteration)

            prog_bar.set_description(f"GAN: loss d: {lossD:.5f} | loss g: {-lossG:.5f}")
            checkpoint.step.assign_add(1)

            ## save image
            if iteration % (save_interval * 10) == 0:
                model_pred_fnoise = netG(fixed_noise)

                backup_para = netG.get_weights()
                netG.set_weights(avg_param_G)

                avg_model_pred_fnoise = netG(fixed_noise)
                netG.set_weights(backup_para)

                gen_imgs = tf.concat(
                    [model_pred_fnoise[0], avg_model_pred_fnoise[0]], axis=0,
                )

                rec_imgs = tf.concat(
                    [tf.image.resize(real_images, [128, 128]), *rec_imgs], axis=0,
                )

                grid_gen = postprocess_images(imgrid(gen_imgs, 8))
                grid_rec = postprocess_images(imgrid(rec_imgs, batch_size))

                kutils.save_img(
                    saved_image_folder + f"/{iteration}.jpg", grid_gen, scale=False
                )
                with summary_writer.as_default():
                    tf.summary.image("Generated images", [grid_gen], step=iteration)
                    tf.summary.image(
                        "Reconstructed images", [grid_rec], step=iteration,
                    )

            ## save weights
            if iteration % (save_interval * 100) == 0 or iteration == total_iterations:
                netG.save_weights(saved_model_folder + f"/generator.h5")
                backup_para = netG.get_weights()
                netG.set_weights(avg_param_G)
                manager.save()
                netG.set_weights(backup_para)

    except KeyboardInterrupt:
        print("RECEIVED KEYBOARD INTERRUPT, SAVING MODEL")
        manager.save()
        netG.save_weights(saved_model_folder + f"/generator.h5")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="region gan")

    parser.add_argument(
        "--path",
        type=str,
        default="../apebase/ipfs",
        help="path of resource dataset, should be a folder that has one or many sub image folders inside",
    )
    parser.add_argument("--name", type=str, default="", help="experiment name")
    parser.add_argument(  # 50000
        "--iter", type=int, default=50000, help="number of iterations"
    )
    parser.add_argument(  # 8
        "--batch_size", type=int, default=8, help="mini batch number of images"
    )
    parser.add_argument(
        "--resume", action="store_true", help="whether to resume training"
    )
    parser.add_argument(  # 'color,translation'
        "--data_aug_policy", type=str, default="color,translation", help=""
    )
    parser.add_argument("--im_size", type=int, default=256, help="image resolution")
    parser.add_argument("--ngf", type=int, default=16, help="")
    parser.add_argument("--ndf", type=int, default=16, help="")
    parser.add_argument("--nz", type=int, default=256, help="")
    parser.add_argument("--nlr", type=float, default=0.0002, help="")
    parser.add_argument("--nbeta1", type=float, default=0.5, help="")
    parser.add_argument("--dataloader_workers", type=int, default=8, help="")
    parser.add_argument("--save_interval", type=int, default=100, help="")
    parser.add_argument("--shuffle_buffer", type=int, default=256, help="")

    args = parser.parse_args()
    print(args)

    train(args)
