'''VGG11/13/16/19 in TensorFlow.'''
import tensorflow as tf


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(tf.keras.Model):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.classifier = tf.keras.layers.Dense(num_classes)
        self.get_features = False

    def call(self, x, training=None):
        features = []
        out = x
        for layer in self.features:
            out = layer(out, training=training)
        features.append(out)
        out = self.pool(out)
        # Flatten the tensor
        out = tf.reshape(out, [tf.shape(out)[0], -1])
        features.append(out)
        out = self.classifier(out)
        if self.get_features:
            return features, out
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers.append(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
            else:
                # Conv2D layer
                layers.append(tf.keras.layers.Conv2D(
                    filters=x,
                    kernel_size=3,
                    padding='same',
                    use_bias=False  # BatchNorm will add bias
                ))
                # BatchNormalization layer
                layers.append(tf.keras.layers.BatchNormalization())
                # ReLU activation
                # Note: TensorFlow doesn't have inplace operations, but the exemptions allow this
                layers.append(tf.keras.layers.ReLU())
                in_channels = x
        return layers


def VGG16(num_classes=10):
    return VGG('VGG16', num_classes=num_classes)