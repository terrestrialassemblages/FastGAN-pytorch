import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow_addons.layers import SpectralNormalization, AdaptiveAveragePooling2D
from einops import rearrange

Blur = lambda: layers.Lambda(lambda x: x)


def conv2d(out_channels, kernel_size, stride=1, padding=0, **kwargs):
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
            Blur(),
            conv2d(out_planes * 2, 3, 1, 1),
            # convTranspose2d(out_planes * 2, 3, 2, 1, use_bias=False),
            NoiseInjection(),
            batchNorm2d(),
            GLU(),
        ],
        name=name,
    )


class SumBranches(layers.Layer):
    def __init__(self, branches):
        super().__init__()
        self.branches = branches

    def call(self, x):
        return sum([f(x) for f in self.branches])


class GlobalContext(layers.Layer):
    def __init__(self, chan_out):
        super().__init__()
        self.chan_out = chan_out

    def build(self, _):
        self.to_k = conv2d(1, 1, stride=1, padding=0)
        chan_intermediate = max(3, self.chan_out // 2)

        self.net = keras.Sequential(
            [
                conv2d(chan_intermediate, 1, stride=1, padding=0),
                layers.LeakyReLU(0.1),
                conv2d(self.chan_out, 1, stride=1, padding=0, activation="sigmoid"),
            ]
        )

    def call(self, x):
        context = self.to_k(x)
        context = tf.nn.softmax(rearrange(context, "b h w c -> b (h w) c"))
        out = tf.einsum(
            "b n i, b n c -> b i c", context, rearrange(x, "b h w c -> b (h w) c")
        )
        out = rearrange(out, "b i c -> b i () c")
        return self.net(out)


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

        self.initial_conv = InitLayer(latent_dim)

        num_layers = int(resolution) - 2
        features = [(n, 2 ** (fmap_inverse_coef - n)) for n in range(2, num_layers + 2)]
        features = [(n[0], min(n[1], fmap_max)) for n in features]
        features = [3 if n[0] >= 8 else n[1] for n in features]

        self.res_layers = range(2, num_layers + 2)
        self.layers_ = []
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
                sle_chan_out = self.res_to_feature_map[residual_layer - 1]

                # if freq_chan_attn:
                #     sle = FCANet(
                #         chan_in=chan_out, chan_out=sle_chan_out, width=2 ** (res + 1)
                #     )
                # else:
                sle = GlobalContext(sle_chan_out)  # SEBlock(sle_chan_out)

            self.layers_.append(
                [UpBlock(chan_out), sle, attn,]
            )

        self.out_conv = conv2d(nchannels, 3, stride=1, padding=1, use_bias=False)

    @tf.function
    def call(self, x):
        x = self.initial_conv(x)
        x = tf.math.l2_normalize(x, axis=3)

        residuals = dict()

        for (res, (up, sle, attn)) in zip(self.res_layers, self.layers_):
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


class SimpleDecoder(layers.Layer):
    def __init__(
        self, chan_out=3, num_upsamples=4,
    ):
        super().__init__()

        self.layers_ = []
        self.final_chan = chan_out
        self.num_upsamples = num_upsamples

    def build(self, input_shape):
        chans = input_shape[-1]

        for ind in range(self.num_upsamples):
            last_layer = ind == (self.num_upsamples - 1)
            chan_out = chans if not last_layer else self.final_chan * 2
            layer = keras.Sequential(
                [
                    layers.UpSampling2D(size=2, interpolation="nearest"),
                    conv2d(chan_out, 3, padding=1),
                    GLU(),
                ]
            )
            self.layers_.append(layer)
            chans //= 2

    def call(self, x):
        for layer in self.layers_:
            x = layer(x)
        return x

@tf.RegisterGradient("ResizeBilinearGrad")
def _ResizeBilinearGrad_grad(op, grad):
    up = tf.image.resize(grad,tf.shape(op.inputs[0])[1:-1])
    return up,None

class Discriminator(keras.Model):
    def __init__(
        self,
        image_size,
        nchannels=3,
        fmap_max=512,
        fmap_inverse_coef=12,
        disc_output_size=5,
        attn_res_layers=[],
    ):
        super().__init__()
        resolution = np.log2(image_size)
        resolution = int(resolution)

        num_non_residual_layers = max(0, int(resolution) - 8)
        num_residual_layers = 8 - 3

        non_residual_resolutions = range(min(8, resolution), 2, -1)
        features = [(n, 2 ** (fmap_inverse_coef - n)) for n in non_residual_resolutions]
        features = [(n[0], min(n[1], fmap_max)) for n in features]

        if num_non_residual_layers == 0:
            res, _ = features[0]
            features[0] = (res, nchannels)

        self.non_residual_layers = []
        for ind in range(num_non_residual_layers):
            first_layer = ind == 0
            last_layer = ind == (num_non_residual_layers - 1)
            chan_out = features[0][-1] if last_layer else nchannels

            self.non_residual_layers.append(
                keras.Sequential(
                    [
                        Blur(),
                        conv2d(chan_out, 4, stride=2, padding=1),
                        layers.LeakyReLU(0.1),
                    ]
                )
            )

        self.residual_layers = []

        for (res, (_, chan_out)) in zip(non_residual_resolutions, features[1:]):
            image_width = 2 ** res

            attn = None
            # if image_width in attn_res_layers:
            #     attn = PreNorm(chan_in, LinearAttention(chan_in))

            self.residual_layers.append(
                [
                    SumBranches(
                        [
                            keras.Sequential(
                                [
                                    Blur(),
                                    conv2d(chan_out, 4, stride=2, padding=1),
                                    layers.LeakyReLU(0.1),
                                    conv2d(chan_out, 3, padding=1),
                                    layers.LeakyReLU(0.1),
                                ]
                            ),
                            keras.Sequential(
                                [
                                    Blur(),
                                    layers.AveragePooling2D(2),
                                    conv2d(chan_out, 1),
                                    layers.LeakyReLU(0.1),
                                ]
                            ),
                        ]
                    ),
                    attn,
                ]
            )

        last_chan = features[-1][-1]
        if disc_output_size == 5:
            self.to_logits = keras.Sequential(
                [conv2d(last_chan, 1), layers.LeakyReLU(0.1), conv2d(1, 4),],
                name="to_logits",
            )
        elif disc_output_size == 1:
            self.to_logits = keras.Sequential(
                [
                    Blur(),
                    conv2d(last_chan, 3, stride=2, padding=1),
                    layers.LeakyReLU(0.1),
                    conv2d(1, 4),
                ],
                name="to_logits",
            )

        self.to_shape_disc_out = keras.Sequential(
            [
                conv2d(64, 3, padding=1),
                # Residual(PreNorm(64, LinearAttention(64))),
                SumBranches(
                    [
                        keras.Sequential(
                            [
                                Blur(),
                                conv2d(32, 4, stride=2, padding=1),
                                layers.LeakyReLU(0.1),
                                conv2d(32, 3, padding=1),
                                layers.LeakyReLU(0.1),
                            ]
                        ),
                        keras.Sequential(
                            [
                                Blur(),
                                layers.AveragePooling2D(2),
                                conv2d(32, 1),
                                layers.LeakyReLU(0.1),
                            ]
                        ),
                    ]
                ),
                # Residual(PreNorm(32, LinearAttention(32))),
                AdaptiveAveragePooling2D((4, 4)),
                conv2d(1, 4),
            ]
        )

        self.decoder1 = SimpleDecoder(chan_out=nchannels)
        self.decoder2 = SimpleDecoder(chan_out=nchannels) if resolution >= 9 else None

        self.loss_fn = keras.losses.MeanSquaredError()

    def call(self, x, calc_aux_loss=False):
        orig_img = x

        for layer in self.non_residual_layers:
            x = layer(x)

        layer_outputs = []

        for (net, attn) in self.residual_layers:
            if attn is not None:
                x = attn(x) + x

            x = net(x)
            layer_outputs.append(x)

        out = rearrange(self.to_logits(x), "b h w c -> b (h w c)")

        img_32x32 = tf.image.resize(orig_img, size=(32, 32))
        out_32x32 = self.to_shape_disc_out(img_32x32)

        if not calc_aux_loss:
            return out, out_32x32, None

        # self-supervised auto-encoding loss

        layer_8x8 = layer_outputs[-1]
        layer_16x16 = layer_outputs[-2]

        recon_img_8x8 = self.decoder1(layer_8x8)
        aux_loss = self.loss_fn(
            recon_img_8x8, tf.image.resize(orig_img, size=recon_img_8x8.shape[1:3])
        )

        if self.decoder2 is not None:
            crop_image_fn = lambda img: rearrange(
                img, "b (m h) (n w) c -> (m n) b h w c", m=2, n=2
            )[tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)]
            img_part, layer_16x16_part = map(crop_image_fn, (orig_img, layer_16x16))

            recon_img_16x16 = self.decoder2(layer_16x16_part)

            aux_loss_16x16 = self.loss_fn(
                recon_img_16x16,
                tf.image.resize(img_part, size=recon_img_16x16.shape[1:3]),
            )

            aux_loss = aux_loss + aux_loss_16x16

        return out, out_32x32, aux_loss
