# import torch
# import torch.nn as nn
# from torch.nn.utils import spectral_norm
# import torch.nn.functional as F

from random import randint
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow_addons.layers import SpectralNormalization
import tensorflow_addons.utils.keras_utils as conv_utils

NFC_MULTI = {
    4: 16,
    8: 8,
    16: 4,
    32: 2,
    64: 2,
    128: 1,
    256: 0.5,
    512: 0.25,
    1024: 0.125,
}


def conv2d(out_channels, kernel_size, stride, padding, **kwargs):
    return SpectralNormalization(
        layers.Conv2D(
            out_channels,
            kernel_size,
            stride,
            "same" if padding else "valid",
            **kwargs,
            kernel_initializer=keras.initializers.RandomNormal(0.0, 0.02),
        )
    )


def batchNorm2d(*args, **kwargs):
    return layers.BatchNormalization(
        *args, **kwargs, gamma_initializer=keras.initializers.RandomNormal(1.0, 0.02),
    )


class DebugPrintLayer(layers.Layer):
    def __init__(self, prefix) -> None:
        super().__init__()
        self.pre = prefix

    def call(self, x):
        print(self.pre, tf.shape(x))
        return x


class PixelNorm(layers.Layer):
    def call(self, input):
        return input * tf.math.rsqrt(
            tf.math.reduce_mean(input ** 2, dim=1, keepdim=True) + 1e-8
        )


class GLU(layers.Layer):
    def build(self, input_shape):
        assert input_shape[-1] % 2 == 0, "Channels must be even."

    def call(self, x):
        nc = tf.shape(x)[-1] // 2
        return x[:, :, :, :nc] * tf.math.sigmoid(x[:, :, :, nc:])


class NoiseInjection(layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, _):
        self.weight = self.add_weight(
            "kernel", shape=(1,), initializer="zeros", trainable=True,
        )

    def call(self, feat, noise=None):
        if noise is None:
            feat_shape = tf.shape(feat)
            noise = tf.random.normal((feat_shape[0], feat_shape[1], feat_shape[2], 1))
        return feat + self.weight * noise


class Swish(layers.Layer):
    def call(self, feat):
        return feat * tf.math.sigmoid(feat)


class SEBlock(layers.Layer):
    def __init__(self, ch_out):
        super().__init__()
        self.main = keras.Sequential(
            [
                AdaptiveAveragePooling2D(4),
                conv2d(ch_out, 4, 1, 0, use_bias=False),
                Swish(),
                conv2d(ch_out, 1, 1, 0, use_bias=False),
                layers.Activation("sigmoid"),
            ]
        )

    def call(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(layers.Layer):
    def __init__(self, channel):
        super().__init__()
        self.init = keras.Sequential(
            [
                layers.Reshape((1, 1, -1)),
                SpectralNormalization(
                    layers.Conv2DTranspose(channel * 2, 4, strides=1, use_bias=False)
                ),
                batchNorm2d(),
                GLU(),
            ]
        )

    def call(self, noise):
        return self.init(noise)


def UpBlock(out_planes):
    block = keras.Sequential(
        [
            layers.UpSampling2D(size=2, interpolation="nearest"),
            conv2d(out_planes * 2, 3, 1, 1, use_bias=False),
            batchNorm2d(),
            GLU(),
        ]
    )
    return block


def UpBlockComp(out_planes):
    block = keras.Sequential(
        [
            layers.UpSampling2D(size=2, interpolation="nearest"),
            conv2d(out_planes * 2, 3, 1, 1, use_bias=False),
            NoiseInjection(),
            batchNorm2d(),
            GLU(),
            conv2d(out_planes * 2, 3, 1, 1, use_bias=False),
            NoiseInjection(),
            batchNorm2d(),
            GLU(),
        ]
    )
    return block


class Generator(keras.Model):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024, *args, **kwargs):
        super(Generator, self).__init__(
            name=f"Generator/ngf_{ngf}/nz_{nz}/nc_{nc}/imsize_{im_size}",
            *args,
            **kwargs,
        )
        self.ngf = ngf
        self.nz = nz
        self.nc = nc
        self.im_size = im_size

    def build(self, _):
        nfc = {k: v * self.ngf for k, v in NFC_MULTI.items()}

        self.init = InitLayer(nfc[4])

        self.feat_8 = UpBlockComp(nfc[8])
        self.feat_16 = UpBlock(nfc[16])
        self.feat_32 = UpBlockComp(nfc[32])
        self.feat_64 = UpBlock(nfc[64])
        self.feat_128 = UpBlockComp(nfc[128])

        self.se_64 = SEBlock(nfc[64])
        self.se_128 = SEBlock(nfc[128])

        self.to_128 = conv2d(self.nc, 1, 1, 0, use_bias=False)
        self.to_big = conv2d(self.nc, 3, 1, 1, use_bias=False)

        if self.im_size > 128:
            self.feat_256 = UpBlock(nfc[256])
            self.se_256 = SEBlock(nfc[256])
        if self.im_size > 256:
            self.feat_512 = UpBlockComp(nfc[512])
            self.se_512 = SEBlock(nfc[512])
        if self.im_size > 512:
            self.feat_1024 = UpBlock(nfc[1024])

    def initialize(self, batch_size: int = 1):
        input_shape = (batch_size, self.nz)
        sample_input = tf.random.normal(shape=input_shape)
        sample_output = self(sample_input)
        return sample_output

    @tf.function
    def call(self, input):
        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)

        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))
        feat_128 = self.se_128(feat_8, self.feat_128(feat_64))
        if self.im_size == 64:
            return [self.to_big(feat_64), self.to_128(feat_128)]
        if self.im_size == 128:
            return [self.to_big(feat_128), self.to_128(feat_128)]

        feat_256 = self.se_256(feat_16, self.feat_256(feat_128))
        if self.im_size == 256:
            return [self.to_big(feat_256), self.to_128(feat_128)]

        feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
        if self.im_size == 512:
            return [self.to_big(feat_512), self.to_128(feat_128)]

        feat_1024 = self.feat_1024(feat_512)

        im_128 = tf.math.tanh(self.to_128(feat_128))
        im_1024 = tf.math.tanh(self.to_big(feat_1024))

        return [im_1024, im_128]


class DownBlock(layers.Layer):
    def __init__(self, out_planes):
        super(DownBlock, self).__init__()

        self.main = keras.Sequential(
            [
                conv2d(out_planes, 4, 2, 1, use_bias=False),
                batchNorm2d(),
                layers.LeakyReLU(alpha=0.2),
            ]
        )

    def call(self, feat):
        return self.main(feat)


class DownBlockComp(layers.Layer):
    def __init__(self, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = keras.Sequential(
            [
                conv2d(out_planes, 4, 2, 1, use_bias=False),
                batchNorm2d(),
                layers.LeakyReLU(alpha=0.2),
                conv2d(out_planes, 3, 1, 1, use_bias=False),
                batchNorm2d(),
                layers.LeakyReLU(alpha=0.2),
            ]
        )

        self.direct = keras.Sequential(
            [
                layers.AveragePooling2D(pool_size=(2, 2)),
                conv2d(out_planes, 1, 1, 0, use_bias=False),
                batchNorm2d(),
                layers.LeakyReLU(alpha=0.2),
            ]
        )

    def call(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(keras.Model):
    def __init__(self, ndf=64, nc=3, im_size=512, *args, **kwargs):
        super(Discriminator, self).__init__(
            name=f"Generator/ndf_{ndf}/nc_{nc}/imsize_{im_size}", *args, **kwargs,
        )
        self.ndf = ndf
        self.nc = nc
        self.im_size = im_size

    def build(self, _):
        nfc = {k: v * self.ndf for k, v in NFC_MULTI.items()}
        for k, v in NFC_MULTI.items():
            nfc[k] = int(v * self.ndf)

        if self.im_size == 1024:
            self.down_from_big = keras.Sequential(
                [
                    conv2d(nfc[1024], 4, 2, 1, use_bias=False),
                    layers.LeakyReLU(alpha=0.2),
                    conv2d(nfc[512], 4, 2, 1, use_bias=False),
                    batchNorm2d(),
                    layers.LeakyReLU(alpha=0.2),
                ]
            )
        elif self.im_size == 512:
            self.down_from_big = keras.Sequential(
                [
                    conv2d(nfc[512], 4, 2, 1, use_bias=False),
                    layers.LeakyReLU(alpha=0.2),
                ]
            )
        elif self.im_size <= 256:
            self.down_from_big = keras.Sequential(
                [
                    conv2d(nfc[512], 3, 1, 1, use_bias=False),
                    layers.LeakyReLU(alpha=0.2),
                ]
            )

        self.down_4 = DownBlockComp(nfc[256])
        self.down_8 = DownBlockComp(nfc[128])
        self.down_16 = DownBlockComp(nfc[64])
        self.down_32 = DownBlockComp(nfc[32])
        self.down_64 = DownBlockComp(nfc[16])

        self.rf_big = keras.Sequential(
            [
                conv2d(nfc[8], 1, 1, 0, use_bias=False),
                batchNorm2d(),
                layers.LeakyReLU(alpha=0.2),
                conv2d(1, 4, 1, 0, use_bias=False),
            ]
        )

        self.se_2_16 = SEBlock(nfc[64])
        self.se_4_32 = SEBlock(nfc[32])
        self.se_8_64 = SEBlock(nfc[16])

        self.down_from_small = keras.Sequential(
            [
                conv2d(nfc[256], 4, 2, 1, use_bias=False),
                layers.LeakyReLU(alpha=0.2),
                DownBlock(nfc[128]),
                DownBlock(nfc[64]),
                DownBlock(nfc[32]),
            ]
        )

        self.rf_small = conv2d(1, 4, 1, 0, use_bias=False)

        self.decoder_big = SimpleDecoder(self.nc)
        self.decoder_part = SimpleDecoder(self.nc)
        self.decoder_small = SimpleDecoder(self.nc)

    def initialize(self, batch_size: int = 1):
        input_shape = (batch_size, self.im_size, self.im_size, 3)
        sample_input = tf.random.uniform(shape=input_shape)
        sample_output = self(sample_input, "real", part=2)
        return sample_output

    # @tf.function
    def call(self, imgs, label, part=2):
        if type(imgs) is not list:
            imgs = [
                tf.image.resize(imgs, [self.im_size, self.im_size], method="nearest"),
                tf.image.resize(imgs, [128, 128], method="nearest"),
            ]

        feat_2 = self.down_from_big(imgs[0])
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)

        feat_16 = self.down_16(feat_8)
        feat_16 = self.se_2_16(feat_2, feat_16)

        feat_32 = self.down_32(feat_16)
        feat_32 = self.se_4_32(feat_4, feat_32)

        feat_last = self.down_64(feat_32)
        feat_last = self.se_8_64(feat_8, feat_last)

        rf_0 = tf.reshape(self.rf_big(feat_last), shape=[-1,])

        feat_small = self.down_from_small(imgs[1])
        rf_1 = tf.reshape(self.rf_small(feat_small), shape=[-1])

        if label == "real":
            rec_img_big = self.decoder_big(feat_last)
            rec_img_small = self.decoder_small(feat_small)

            rec_img_part = None
            stop_idx = 8 if self.im_size >= 256 else self.im_size // 32
            if part == 0:
                rec_img_part = self.decoder_part(feat_32[:, :stop_idx, :stop_idx, :])
            if part == 1:
                rec_img_part = self.decoder_part(feat_32[:, :stop_idx, stop_idx:, :])
            if part == 2:
                rec_img_part = self.decoder_part(feat_32[:, stop_idx:, :stop_idx, :])
            if part == 3:
                rec_img_part = self.decoder_part(feat_32[:, stop_idx:, stop_idx:, :])

            return (
                tf.concat([rf_0, rf_1], axis=0),
                [rec_img_big, rec_img_small, rec_img_part],
            )

        return tf.concat([rf_0, rf_1], axis=0)


class SimpleDecoder(layers.Layer):
    """docstring for CAN_SimpleDecoder"""

    def __init__(self, nc=3):
        super(SimpleDecoder, self).__init__()
        nfc = {}
        for k, v in NFC_MULTI.items():
            nfc[k] = int(v * 32)

        self.main = keras.Sequential(
            [
                AdaptiveAveragePooling2D(8),
                UpBlock(nfc[16]),
                UpBlock(nfc[32]),
                UpBlock(nfc[64]),
                UpBlock(nfc[128]),
                conv2d(nc, 3, 1, 1, use_bias=False),
                layers.Activation("tanh"),
            ]
        )

    def call(self, input):
        # input shape: c x 4 x 4
        return self.main(input)


# https://github.com/tensorflow/addons/pull/2322
class AdaptiveAveragePooling2D(tf.keras.layers.Layer):
    def __init__(
        self, output_size, **kwargs,
    ):
        self.reduce_function = tf.reduce_mean
        self.output_size = (output_size, output_size)
        self.output_size_x, self.output_size_y = self.output_size

        super().__init__(**kwargs)

    def call(self, inputs, *args):
        start_points_x = tf.cast(
            (
                tf.range(self.output_size_x, dtype=tf.float32)
                * tf.cast((tf.shape(inputs)[1] / self.output_size_x), tf.float32)
            ),
            tf.int32,
        )
        end_points_x = tf.cast(
            tf.math.ceil(
                (
                    (tf.range(self.output_size_x, dtype=tf.float32) + 1)
                    * tf.cast((tf.shape(inputs)[1] / self.output_size_x), tf.float32)
                )
            ),
            tf.int32,
        )

        start_points_y = tf.cast(
            (
                tf.range(self.output_size_y, dtype=tf.float32)
                * tf.cast((tf.shape(inputs)[2] / self.output_size_y), tf.float32)
            ),
            tf.int32,
        )
        end_points_y = tf.cast(
            tf.math.ceil(
                (
                    (tf.range(self.output_size_y, dtype=tf.float32) + 1)
                    * tf.cast((tf.shape(inputs)[2] / self.output_size_y), tf.float32)
                )
            ),
            tf.int32,
        )
        pooled = []
        for idx in range(self.output_size_x):
            pooled.append(
                self.reduce_function(
                    inputs[:, start_points_x[idx] : end_points_x[idx], :, :],
                    axis=1,
                    keepdims=True,
                )
            )
        x_pooled = tf.concat(pooled, axis=1)

        pooled = []
        for idx in range(self.output_size_y):
            pooled.append(
                self.reduce_function(
                    x_pooled[:, :, start_points_y[idx] : end_points_y[idx], :],
                    axis=2,
                    keepdims=True,
                )
            )
        y_pooled = tf.concat(pooled, axis=2)
        return y_pooled

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        shape = tf.TensorShape(
            [input_shape[0], self.output_size[0], self.output_size[1], input_shape[3],]
        )

        return shape

    def get_config(self):
        config = {
            "output_size": self.output_size,
        }
        base_config = super().get_config()
        return {**base_config, **config}
