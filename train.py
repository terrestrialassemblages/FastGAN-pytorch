import tensorflow as tf
import tensorflow.keras.utils as kutils
from tensorflow.keras import optimizers
from tensorflow.keras import layers

import argparse
import random
from tqdm import tqdm

import lpips_tf
from models import Discriminator, Generator
from operation import get_dir, imgrid
from diffaug import DiffAugment

ploss = lpips_tf.PerceptualLoss(model="net-lin", net="alex")
# normalize = layers.Normalization(mean=[0.5, 0.5, 0.5], variance=[0.5, 0.5, 0.5])


def get_perceptual_loss(input0, input1):
    return tf.math.reduce_sum(ploss(input0, input1))


def transform_fn(image):
    image = tf.image.random_flip_left_right(image)
    # image = normalize(image)
    image = image / 255.0
    return image


def crop_image_by_part(image, part):
    hw = image.shape[1] // 2
    if part == 0:
        return image[:, :hw, :hw, :]
    if part == 1:
        return image[:, :hw, hw:, :]
    if part == 2:
        return image[:, hw:, :hw, :]
    if part == 3:
        return image[:, hw:, hw:, :]


@tf.function()
def train_step_d1(net, data, label="real"):
    """Train function of discriminator"""
    if label == "real":
        with tf.GradientTape() as tape:
            part = random.randint(0, 3)
            pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
            err = (
                tf.nn.relu(
                    tf.math.reduce_mean(
                        tf.random.uniform(pred.shape) * 0.2 + 0.8 - pred
                    )
                )
                + get_perceptual_loss(
                    rec_all, tf.image.resize(data, (rec_all.shape[1], rec_all.shape[1]))
                )
                + get_perceptual_loss(
                    rec_small,
                    tf.image.resize(data, (rec_small.shape[1], rec_small.shape[1])),
                )
                + get_perceptual_loss(
                    rec_part,
                    tf.image.resize(
                        crop_image_by_part(data, part),
                        (rec_part.shape[1], rec_part.shape[1]),
                    ),
                )
            )
        return (
            err, tape.gradient(err, net.trainable_weights),
            tf.math.reduce_mean(pred),
            (rec_all, rec_small, rec_part),
        )
    else:
        with tf.GradientTape() as tape:
            pred = net(data, label)
            err = tf.math.reduce_mean(
                tf.nn.relu(tf.random.uniform(pred.shape) * 0.2 + 0.8 + pred)
            )
        return err, tape.gradient(err, net.trainable_weights), tf.math.reduce_mean(pred)


@tf.function()
def train_step_d2(net, data, label="real"):
    """Train function of discriminator"""
    if label == "real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label="real", part=part)
        err = (
            tf.nn.relu(
                tf.math.reduce_mean(tf.random.uniform(pred.shape) * 0.2 + 0.8 - pred)
            )
            + get_perceptual_loss(
                rec_all, tf.image.resize(data, (rec_all.shape[1], rec_all.shape[1]))
            )
            + get_perceptual_loss(
                rec_small,
                tf.image.resize(data, (rec_small.shape[1], rec_small.shape[1])),
            )
            + get_perceptual_loss(
                rec_part,
                tf.image.resize(
                    crop_image_by_part(data, part),
                    (rec_part.shape[1], rec_part.shape[1]),
                ),
            )
        )
        return err, tf.math.reduce_mean(pred), (rec_all, rec_small, rec_part)

    else:
        pred = net(data, label="fake")
        err = tf.math.reduce_mean(
            tf.nn.relu(tf.random.uniform(pred.shape) * 0.2 + 0.8 + pred)
        )
        return err, tf.math.reduce_mean(pred)


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
    # current_iteration = args.current_iteration
    save_interval = args.save_interval
    policy = args.data_aug_policy
    shuffle_buffer = args.shuffle_buffer

    saved_model_folder, saved_image_folder, log_folder = get_dir(args)
    summary_writer = tf.summary.create_file_writer(log_folder)

    ## load dataset and apply transforms
    ds = kutils.image_dataset_from_directory(
        data_root, image_size=(im_size, im_size), labels=None, batch_size=None
    )
    ds = (
        ds.map(transform_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .repeat()
        .shuffle(shuffle_buffer)
    )
    # ds = ds.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    dataloader = iter(ds)

    ## load networks and checkpoints
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netD = Discriminator(ndf=ndf, im_size=im_size)

    # forward passes to build model
    netG.initialize(batch_size)
    netD.initialize(batch_size)

    fixed_noise = tf.random.normal((16, nz), 0, 1, seed=42)

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
        for _ in (prog_bar := tqdm(range(current_iteration, total_iterations + 1))) :
            checkpoint.step.assign_add(1)
            iteration = checkpoint.step.numpy()

            real_images = next(dataloader)
            real_images = DiffAugment(real_images, policy=policy)

            current_batch_size = real_images.shape[0]
            
            
            noise = tf.random.normal((current_batch_size, nz), 0, 1)

            fake_images = netG(noise)
            fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]

            # # Train the discriminator.
            loss_d_real, grads_dr, err_dr, rec_img = train_step_d1(netD, real_images, label="real")
            loss_d_fake, grads_df, err_df = train_step_d1(netD, fake_images, label="fake")
            # optimizerD.apply_gradients(zip(grads_dr+grads_df, netD.trainable_weights))
            
            optimizerD.apply_gradients(
                (grad, var)
                for (grad, var) in zip(grads_dr, netD.trainable_weights)
                if grad is not None
            )
            optimizerD.apply_gradients(
                (grad, var)
                for (grad, var) in zip(grads_df, netD.trainable_weights)
                if grad is not None
            )

            # 2. train loop
            # with tf.GradientTape(persistent=True) as tape:
            #     loss_d_real, err_dr, rec_img = train_step_d2(netD, real_images, label="real")
            #     loss_d_fake, err_df = train_step_d2(netD, fake_images, label="fake")
            # grads = tape.gradient(loss_d_real, netD.trainable_weights) + tape.gradient(
            #     loss_d_fake, netD.trainable_weights
            # )
            # optimizerD.apply_gradients(zip(grads, netD.trainable_weights))
            
            with tf.GradientTape() as tape:
                fake_images = netG(noise)
                fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
                pred_g = netD(fake_images, "fake")
                loss_g = -tf.math.reduce_mean(pred_g)
            grads = tape.gradient(loss_g, netG.trainable_weights)
            optimizerG.apply_gradients(zip(grads, netG.trainable_weights))

            for i, (w, avg_w) in enumerate(zip(netG.get_weights(), avg_param_G)):
                avg_param_G[i] = avg_w * 0.999 + 0.001 * w

            with summary_writer.as_default():
                tf.summary.scalar("Pred/DiscriminatorReal", err_dr, step=iteration)
                tf.summary.scalar("Pred/DiscriminatorFake", err_df, step=iteration)
                # tf.summary.scalar("Pred/Generator", loss_g, step=iteration)

                tf.summary.scalar("Loss/DiscriminatorReal", loss_d_real, step=iteration)
                tf.summary.scalar("Loss/DiscriminatorFake", loss_d_fake, step=iteration)
                tf.summary.scalar("Loss/Generator", loss_g, step=iteration)
            prog_bar.set_description(
                f"GAN: loss d: {err_dr:.5f} | loss g: {-loss_g:.5f}"
            )

            ## save image
            if iteration % (save_interval * 10) == 0:
                model_pred_fnoise = netG(fixed_noise)

                backup_para = netG.get_weights()
                netG.set_weights(avg_param_G)

                avg_model_pred_fnoise = netG(fixed_noise)
                netG.set_weights(backup_para)

                gen_imgs = tf.concat(
                    [
                        model_pred_fnoise[0],
                        avg_model_pred_fnoise[0],
                    ],
                    axis=0,
                )

                rec_imgs = tf.concat(
                    [tf.image.resize(real_images, (128, 128)), *rec_img,], axis=0,
                )

                grid_gen = imgrid((gen_imgs + 1) * 127.5, 8)
                grid_rec = imgrid((rec_imgs + 1) * 127.5, 8)

                kutils.save_img(saved_image_folder + f"/{iteration}.jpg", grid_gen)
                with summary_writer.as_default():
                    tf.summary.image(
                        "Generated images", tf.expand_dims(grid_gen, 0), step=iteration
                    )
                    tf.summary.image(
                        "Reconstructed images",
                        tf.expand_dims(grid_rec, 0),
                        step=iteration,
                    )

            ## save weights
            if iteration % (save_interval * 100) == 0 or iteration == total_iterations:
                backup_para = netG.get_weights()
                netG.set_weights(avg_param_G)
                manager.save()
                netG.save(saved_model_folder + f"/{args.name}")
                netG.set_weights(backup_para)

    except KeyboardInterrupt:
        print("RECEIVED KEYBOARD INTERRUPT, SAVING MODEL")
        manager.save()
        netG.save(saved_model_folder + f"/{args.name}")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="region gan")

    parser.add_argument(
        "--path",
        type=str,
        default="../apebase/ipfs",
        help="path of resource dataset, should be a folder that has one or many sub image folders inside",
    )
    parser.add_argument(
        "--name", type=str, default="nf16_nz256_imsize256", help="experiment name"
    )
    parser.add_argument(  # 50000
        "--iter", type=int, default=50000, help="number of iterations"
    )
    parser.add_argument(  # 8
        "--batch_size", type=int, default=8, help="mini batch number of images"
    )
    parser.add_argument(
        "--resume", type=bool, default=False, help="whether to resume training"
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
