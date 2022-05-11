import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.applications import vgg16
from ..models import (
    conv2d,
    batchNorm2d,
    InitLayer,
    UpBlockComp,
    UpBlock,
    SEBlock,
    DownBlockComp,
    SimpleDecoder,
)

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

        self.to_big = conv2d(self.nc, 3, 1, 1, use_bias=False, activation="tanh")

        if self.im_size > 128:
            self.feat_256 = UpBlock(nfc[256], name="feat_256")
            self.se_256 = SEBlock(nfc[256])
        if self.im_size > 256:
            self.feat_512 = UpBlockComp(nfc[512], name="feat_512")
            self.se_512 = SEBlock(nfc[512])
        if self.im_size > 512:
            self.feat_1024 = UpBlock(nfc[1024], name="feat_1024")

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

    @tf.function
    def call(self, img):
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

        return rf_logits, rec_img_big


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
