# import torch
# import torch.nn as nn
# from torch.nn.utils import spectral_norm
# import torch.nn.functional as F

from random import randint
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow_addons.layers import SpectralNormalization, AdaptiveAveragePooling2D

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
    def __init__(self, number) -> None:
        super().__init__()
        self.num = number

    def call(self, x):
        print("!" * self.num, x.shape)
        return x


class PixelNorm(layers.Layer):
    def call(self, input):
        return input * tf.math.rsqrt(
            tf.math.reduce_mean(input ** 2, dim=1, keepdim=True) + 1e-8
        )


class Reshape(layers.Layer):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def call(self, feat):
        batch = feat.shape[0]
        return tf.reshape(feat, [batch, *self.target_shape])


class GLU(layers.Layer):
    def build(self, input_shape):
        assert input_shape[-1] % 2 == 0, "Channels must be even."

    def call(self, x):
        nc = x.shape[-1] // 2
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
            batch, _, height, width = feat.shape
            noise = tf.random.normal((batch, 1, height, width))

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
                SpectralNormalization(
                    layers.Conv2DTranspose(channel * 2, 4, strides=1, use_bias=False)
                ),
                batchNorm2d(),
                GLU(),
            ]
        )

    def call(self, noise):
        noise = tf.reshape(noise, [noise.shape[0], 1, 1, -1])
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
            name=f"Generator/ngf={ngf}/nz={nz}/nc={nc}/imsize={im_size}",
            *args,
            **kwargs,
        )
        self.ngf = ngf
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
            name=f"Generator/ndf={ndf}/nc={nc}/imsize={im_size}", *args, **kwargs,
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

    def call(self, imgs, label, part=None):
        if type(imgs) is not list:
            imgs = [
                tf.image.resize(imgs, (self.im_size, self.im_size), method="nearest"),
                tf.image.resize(imgs, (128, 128), method="nearest"),
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

        rf_0 = tf.reshape(self.rf_big(feat_last), [-1,])

        feat_small = self.down_from_small(imgs[1])
        rf_1 = tf.reshape(self.rf_small(feat_small), [-1])

        if label == "real":
            rec_img_big = self.decoder_big(feat_last)
            rec_img_small = self.decoder_small(feat_small)

            assert part is not None
            rec_img_part = None
            if part == 0:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, :8])
            if part == 1:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, 8:])
            if part == 2:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, :8])
            if part == 3:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, 8:])

            return (
                tf.concat([rf_0, rf_1], axis=0),
                [rec_img_big, rec_img_small, rec_img_part],
            )

        return tf.concat([rf_0, rf_1], axis=0)


class SimpleDecoder(layers.Layer):
    """docstring for CAN_SimpleDecoder"""

    def __init__(self, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {
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
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * 32)

        def upBlock(out_planes):
            block = keras.Sequential(
                [
                    layers.UpSampling2D(size=2, interpolation="nearest"),
                    conv2d(out_planes * 2, 3, 1, 1, use_bias=False),
                    batchNorm2d(),
                    GLU(),
                ]
            )
            return block

        self.main = keras.Sequential(
            [
                AdaptiveAveragePooling2D(8),
                upBlock(nfc[16]),
                upBlock(nfc[32]),
                upBlock(nfc[64]),
                upBlock(nfc[128]),
                conv2d(nc, 3, 1, 1, use_bias=False),
                layers.Activation("tanh"),
            ]
        )

    def call(self, input):
        # input shape: c x 4 x 4
        return self.main(input)
