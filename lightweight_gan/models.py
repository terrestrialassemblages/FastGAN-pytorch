import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow_addons.layers import SpectralNormalization, AdaptiveAveragePooling2D


def conv2d(out_channels, kernel_size, stride, padding, **kwargs):
    # return SpectralNormalization(
    return layers.Conv2D(
        out_channels,
        kernel_size,
        stride,
        "same" if padding else "valid",
        **kwargs,
        kernel_initializer="orthogonal",  # keras.initializers.RandomNormal(0.0, 0.02),
    )  # )


def convTranspose2d(out_channels, kernel_size, stride, padding, **kwargs):
    # return SpectralNormalization(
    return layers.Conv2DTranspose(
        out_channels,
        kernel_size,
        stride,
        "same" if padding else "valid",
        **kwargs,
        kernel_initializer="orthogonal",  # keras.initializers.RandomNormal(0.0, 0.02),
    )  # )


def batchNorm2d(*args, **kwargs):
    return layers.BatchNormalization(
        *args, **kwargs, gamma_initializer=keras.initializers.RandomNormal(1.0, 0.02),
    )


class GLU(layers.Layer):
    def build(self, input_shape):
        assert input_shape[-1] % 2 == 0, "Channels must be even."

    def call(self, x):
        nc = tf.shape(x)[-1] // 2
        return x[:, :, :, :nc] * tf.math.sigmoid(x[:, :, :, nc:])


def InitLayer(channel, name=None):
    return keras.Sequential(
        [
            layers.Reshape((1, 1, -1)),
            convTranspose2d(channel * 2, 4, 1, 0),
            batchNorm2d(),
            GLU(),
        ],
        name=name,
    )


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
                conv2d(self.ch_out, 4, 1, 0),
                layers.LeakyReLU(alpha=0.1),  # Swish(),
                conv2d(self.ch_out, 1, 1, 0, activation="sigmoid"),
            ]
        )

    def summary(self):
        self.main.summary()

    def call(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


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


def UpBlock(out_planes, name=None):
    return keras.Sequential(
        [
            layers.UpSampling2D(size=2, interpolation="nearest"),
            conv2d(out_planes * 2, 3, 1, 1, use_bias=False),
            # convTranspose2d(out_planes * 2, 3, 2, 1, use_bias=False),
            NoiseInjection(),
            batchNorm2d(),
            GLU(),
        ],
        name=name,
    )


class Generator(keras.Model):
    def __init__(
        self,
        image_size,
        nchannels=3,
        latent_dim=256,
        fmap_max=512,
        fmap_inverse_coef=12,
        attn_res_layers=[],
        freq_chan_attn=False,
    ):
        super().__init__()
        resolution = np.log2(image_size)

        self.initial_conv = InitLayer()

        num_layers = int(resolution) - 2
        features = [(n, 2 ** (fmap_inverse_coef - n)) for n in range(2, num_layers + 2)]
        features = [(n[0], min(n[1], fmap_max)) for n in features]
        features = [3 if n[0] >= 8 else n[1] for n in features]

        self.res_layers = range(2, num_layers + 2)
        self.layers = keras.Sequential()
        self.res_to_feature_map = dict(zip(self.res_layers, features))

        self.sle_map = ((3, 7), (4, 8), (5, 9), (6, 10))
        self.sle_map = {
            t[0]: t[1]
            for t in self.sle_map
            if t[0] <= resolution and t[1] <= resolution
        }

        for (res, chan_out) in zip(self.res_layers, features):

            attn = None
            # image_width = 2 ** res
            # if image_width in attn_res_layers:
            #     attn = PreNorm(chan_in, LinearAttention(chan_in))

            sle = None
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_chan_out = self.res_to_feature_map[residual_layer - 1][-1]

                # if freq_chan_attn:
                #     sle = FCANet(
                #         chan_in=chan_out, chan_out=sle_chan_out, width=2 ** (res + 1)
                #     )
                # else:
                #     GlobalContext(chan_in=chan_out, chan_out=sle_chan_out)
                sle = SEBlock(sle_chan_out)

            self.layers.append(
                [UpBlock(chan_out), sle, attn,]
            )

        self.out_conv = conv2d(
            nchannels, 3, 1, 1, use_bias=False
        )

    @tf.function
    def call(self, x):
        x = self.initial_conv(x)
        x = tf.math.l2_normalize(x, axis=1)

        residuals = dict()

        for (res, (up, sle, attn)) in zip(self.res_layers, self.layers):
            if attn is not None:
                x = attn(x) + x

            x = up(x)

            if sle is not None:
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            next_res = res + 1
            if next_res in residuals:
                x = x * residuals[next_res]

        return self.out_conv(x)
