import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow_addons.layers import SpectralNormalization, AdaptiveAveragePooling2D
from tensorflow.keras.applications import vgg16

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
            kernel_initializer="orthogonal",  # keras.initializers.RandomNormal(0.0, 0.02),
        )
    )


def convTranspose2d(out_channels, kernel_size, stride, padding, **kwargs):
    return SpectralNormalization(
        layers.Conv2DTranspose(
            out_channels,
            kernel_size,
            stride,
            "same" if padding else "valid",
            **kwargs,
            kernel_initializer="orthogonal",  # keras.initializers.RandomNormal(0.0, 0.02),
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
    def build(self, _):
        self.weight = self.add_weight(
            "kernel", shape=(1,), initializer="zeros", trainable=True,
        )

    def call(self, feat, noise=None):
        if noise is None:
            feat_shape = tf.shape(feat)
            noise = tf.random.normal(
                (feat_shape[0], feat_shape[1], feat_shape[2], 1),
                dtype=self.compute_dtype,
            )
        return feat + self.weight * noise


class Swish(layers.Layer):
    def call(self, feat):
        return feat * tf.math.sigmoid(feat)


class SEBlock(layers.Layer):
    def __init__(self, ch_out):
        super().__init__()
        self.ch_out = ch_out

    def build(self, _):
        self.main = keras.Sequential(
            [
                AdaptiveAveragePooling2D(4),
                conv2d(self.ch_out, 4, 1, 0, use_bias=False),
                Swish(),
                conv2d(self.ch_out, 1, 1, 0, use_bias=False),
                layers.Activation("sigmoid"),
            ]
        )

    def summary(self):
        self.main.summary()

    def call(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


def InitLayer(channel, name=None):
    return keras.Sequential(
        [
            layers.Reshape((1, 1, -1)),
            convTranspose2d(channel * 2, 4, 1, 0, use_bias=False),
            batchNorm2d(),
            GLU(),
        ],
        name=name,
    )


def UpBlock(out_planes, name=None):
    return keras.Sequential(
        [
            # layers.UpSampling2D(size=2, interpolation="nearest"),
            # conv2d(out_planes * 2, 3, 1, 1, use_bias=False),
            convTranspose2d(out_planes * 2, 3, 2, 1, use_bias=False),
            batchNorm2d(),
            GLU(),
        ],
        name=name,
    )


def UpBlockComp(out_planes, name=None):
    return keras.Sequential(
        [
            # layers.UpSampling2D(size=2, interpolation="nearest"),
            # conv2d(out_planes * 2, 3, 1, 1, use_bias=False),
            convTranspose2d(out_planes * 2, 3, 2, 1, use_bias=False),
            NoiseInjection(),
            batchNorm2d(),
            GLU(),
            conv2d(out_planes * 2, 3, 1, 1, use_bias=False),
            NoiseInjection(),
            batchNorm2d(),
            GLU(),
        ],
        name=name,
    )


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

        self.init = InitLayer(nfc[4], name="init")

        self.feat_8 = UpBlockComp(nfc[8], name="feat_8")
        self.feat_16 = UpBlock(nfc[16], name="feat_16")
        self.feat_32 = UpBlockComp(nfc[32], name="feat_32")
        self.feat_64 = UpBlock(nfc[64], name="feat_64")
        self.feat_128 = UpBlockComp(nfc[128], name="feat_128")

        self.se_64 = SEBlock(nfc[64])
        self.se_128 = SEBlock(nfc[128])

        self.to_big = keras.Sequential(
            [conv2d(self.nc, 3, 1, 1, use_bias=False), layers.Activation("tanh")],
            name="to_big",
        )

        if self.im_size > 128:
            self.feat_256 = UpBlock(nfc[256], name="feat_256")
            self.se_256 = SEBlock(nfc[256])
            
        if self.im_size > 256:
            self.feat_512 = UpBlockComp(nfc[512], name="feat_512")
            self.se_512 = SEBlock(nfc[512])
            
        if self.im_size > 512:
            self.feat_1024 = UpBlock(nfc[1024], name="feat_1024")

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
        if self.im_size == 64:
            return self.to_big(feat_64)
        feat_128 = self.se_128(feat_8, self.feat_128(feat_64))
        if self.im_size == 128:
            return self.to_big(feat_128)
        feat_256 = self.se_256(feat_16, self.feat_256(feat_128))
        if self.im_size == 256:
            return self.to_big(feat_256)
        feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
        if self.im_size == 512:
            return self.to_big(feat_512)
        feat_1024 = self.feat_1024(feat_512)
        return self.to_big(feat_1024)


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
                layers.Flatten(),
            ]
        )

        self.se_2_16 = SEBlock(nfc[64])
        self.se_4_32 = SEBlock(nfc[32])
        self.se_8_64 = SEBlock(nfc[16])

        self.decoder_big = SimpleDecoder(self.nc)
        self.decoder_part = SimpleDecoder(self.nc)

    def initialize(self, batch_size: int = 1):
        input_shape = (batch_size, self.im_size, self.im_size, 3)
        sample_input = tf.random.uniform(shape=input_shape)
        sample_output = self(sample_input, "real", part=2)
        return sample_output

    @tf.function
    def call(self, img, part):
        feat_2 = self.down_from_big(img)
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)

        feat_16 = self.down_16(feat_8)
        feat_16 = self.se_2_16(feat_2, feat_16)

        feat_32 = self.down_32(feat_16)
        feat_32 = self.se_4_32(feat_4, feat_32)

        feat_last = self.down_64(feat_32)
        feat_last = self.se_8_64(feat_8, feat_last)

        rf_logits = self.rf_big(feat_last)

        rec_img_big = self.decoder_big(feat_last)

        stop_idx = 8

        rec_img_part = tf.switch_case(
            part,
            branch_fns={
                0: lambda: self.decoder_part(feat_32[:, :stop_idx, :stop_idx, :]),
                1: lambda: self.decoder_part(feat_32[:, :stop_idx, stop_idx:, :]),
                2: lambda: self.decoder_part(feat_32[:, stop_idx:, :stop_idx, :]),
                3: lambda: self.decoder_part(feat_32[:, stop_idx:, stop_idx:, :]),
            },
        )

        return rf_logits, [rec_img_big, rec_img_part]


# https://github.com/tensorflow/addons/pull/2322
# class AdaptiveAveragePooling2D(tf.keras.layers.Layer):
#     def __init__(
#         self, output_size, **kwargs,
#     ):
#         self.reduce_function = tf.reduce_mean
#         self.output_size = (output_size, output_size)
#         self.output_size_x, self.output_size_y = self.output_size

#         super().__init__(**kwargs)

#     def call(self, inputs, *args):
#         start_points_x = tf.cast(
#             (
#                 tf.range(self.output_size_x, dtype=tf.float32)
#                 * tf.cast((tf.shape(inputs)[1] / self.output_size_x), tf.float32)
#             ),
#             tf.int32,
#         )
#         end_points_x = tf.cast(
#             tf.math.ceil(
#                 (
#                     (tf.range(self.output_size_x, dtype=tf.float32) + 1)
#                     * tf.cast((tf.shape(inputs)[1] / self.output_size_x), tf.float32)
#                 )
#             ),
#             tf.int32,
#         )

#         start_points_y = tf.cast(
#             (
#                 tf.range(self.output_size_y, dtype=tf.float32)
#                 * tf.cast((tf.shape(inputs)[2] / self.output_size_y), tf.float32)
#             ),
#             tf.int32,
#         )
#         end_points_y = tf.cast(
#             tf.math.ceil(
#                 (
#                     (tf.range(self.output_size_y, dtype=tf.float32) + 1)
#                     * tf.cast((tf.shape(inputs)[2] / self.output_size_y), tf.float32)
#                 )
#             ),
#             tf.int32,
#         )
#         pooled = []
#         for idx in range(self.output_size_x):
#             pooled.append(
#                 self.reduce_function(
#                     inputs[:, start_points_x[idx] : end_points_x[idx], :, :],
#                     axis=1,
#                     keepdims=True,
#                 )
#             )
#         x_pooled = tf.concat(pooled, axis=1)

#         pooled = []
#         for idx in range(self.output_size_y):
#             pooled.append(
#                 self.reduce_function(
#                     x_pooled[:, :, start_points_y[idx] : end_points_y[idx], :],
#                     axis=2,
#                     keepdims=True,
#                 )
#             )
#         y_pooled = tf.concat(pooled, axis=2)
#         return y_pooled

#     def compute_output_shape(self, input_shape):
#         input_shape = tf.TensorShape(input_shape).as_list()
#         shape = tf.TensorShape(
#             [input_shape[0], self.output_size[0], self.output_size[1], input_shape[3],]
#         )

#         return shape

#     def get_config(self):
#         config = {
#             "output_size": self.output_size,
#         }
#         base_config = super().get_config()
#         return {**base_config, **config}


class LpipsNetwork(keras.Model):
    def __init__(
        self, content_layers=["block1_conv2", "block2_conv2", "block3_conv3"],
    ):
        super(LpipsNetwork, self).__init__()

        vgg = vgg16.VGG16(include_top=False, weights="imagenet")
        vgg.trainable = False
        model_outputs = [vgg.get_layer(name).output for name in content_layers]
        self.model = tf.keras.models.Model(vgg.input, model_outputs)

    def _deprocess(self, img):
        return (img * 127.5) + 127.5

    @tf.function
    def call(self, real_img, rec_img):
        real_img = self._deprocess(real_img)
        real_img = vgg16.preprocess_input(real_img)
        real_maps = self.model(real_img)

        rec_img = self._deprocess(rec_img)
        rec_img = vgg16.preprocess_input(rec_img)
        rec_maps = self.model(rec_img)

        loss = tf.add_n(
            [
                tf.reduce_mean(tf.keras.losses.MAE(real, rec))
                for real, rec in zip(real_maps, rec_maps)
            ]
        )
        return loss
