import tensorflow as tf
from tensorflow import keras
import numpy as np

def get_efficientnetb0_backbone():
    """Builds EfficientNetB0 with pre-trained imagenet weights"""
    backbone = keras.applications.EfficientNetB0(
        include_top=False, input_shape=[None, None, 3], weights='imagenet'
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["block4a_expand_activation", "block6a_expand_activation", "top_activation"]
    ]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )

class BiFeaturePyramid(keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
    """

    def __init__(self, backbone=None, channels=64, **kwargs):
        super(BiFeaturePyramid, self).__init__(name="BiFeaturePyramid", **kwargs)
        self.channels = channels
        self.backbone = backbone if backbone else get_efficientnetb0_backbone()
        self.conv_c3_1x1 = keras.layers.Conv2D(self.channels, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(self.channels, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(self.channels, 1, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(self.channels, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(self.channels, 3, 2, "same")

        self.upsample_2x = keras.layers.UpSampling2D(2)
        self.maxpool_2x = tf.keras.layers.MaxPool2D()
        self.layers = [[], [], []]

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))

        def bifpn(p3, p4, p5, p6, p7, i):

            if len(self.layers[i]) == 0:
                self.layers[i].append(keras.layers.Conv2D(self.channels, 3, 1, "same", activation="relu"))
                self.layers[i].append(keras.layers.Conv2D(self.channels, 3, 1, "same", activation="relu"))
                self.layers[i].append(keras.layers.Conv2D(self.channels, 3, 1, "same", activation="relu"))

                self.layers[i].append(keras.layers.Conv2D(self.channels, 3, 1, "same", activation="relu"))
                self.layers[i].append(keras.layers.Conv2D(self.channels, 3, 1, "same", activation="relu"))
                self.layers[i].append(keras.layers.Conv2D(self.channels, 3, 1, "same", activation="relu"))
                self.layers[i].append(keras.layers.Conv2D(self.channels, 3, 1, "same", activation="relu"))
                self.layers[i].append(keras.layers.Conv2D(self.channels, 3, 1, "same", activation="relu"))

            sl = self.layers[i]

            _p6 = self.upsample_2x(p7) + p6
            _p5 = self.upsample_2x(_p6) + p5
            _p4 = self.upsample_2x(_p5) + p4
            p3 = self.upsample_2x(_p4) + p3
            p4 = sl[0](_p4) + self.maxpool_2x(p3) + p4
            p5 = sl[1](_p5) + self.maxpool_2x(p4) + p5
            p6 = sl[2](_p6) + self.maxpool_2x(p5) + p6
            p7 = p7 + self.maxpool_2x(p6)

            return sl[3](p3), sl[4](p4), sl[5](p5), sl[6](p6), sl[7](p7)

        p3_output, p4_output, p5_output, p6_output, p7_output = bifpn(p3_output, p4_output, p5_output, p6_output, p7_output, 0)
        p3_output, p4_output, p5_output, p6_output, p7_output = bifpn(p3_output, p4_output, p5_output, p6_output, p7_output, 1)
        p3_output, p4_output, p5_output, p6_output, p7_output = bifpn(p3_output, p4_output, p5_output, p6_output, p7_output, 2)

        return p3_output, p4_output, p5_output, p6_output, p7_output

def build_head(output_filters, bias_init, channels):
    """Builds the class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.

    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = keras.Sequential([keras.Input(shape=[None, None, channels])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            keras.layers.Conv2D(channels, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(keras.layers.ReLU())
    head.add(
        keras.layers.Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head

class EfficientDet(keras.Model):
    """A subclassed Keras model implementing the EfficientDet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
    """

    def __init__(self, num_classes, backbone=None, **kwargs):
        super(EfficientDet, self).__init__(name="EfficientDet", **kwargs)
        self.channels = 64
        self.fpn = BiFeaturePyramid(backbone, self.channels)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability, self.channels)
        self.box_head = build_head(9 * 4, "zeros", self.channels)

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)