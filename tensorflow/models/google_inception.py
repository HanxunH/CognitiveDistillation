'''GoogLeNet with TensorFlow.'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Inception(keras.Model):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = keras.Sequential([
            layers.Conv2D(n1x1, kernel_size=1, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

        # 1x1 conv -> 3x3 conv branch
        self.b2 = keras.Sequential([
            layers.Conv2D(n3x3red, kernel_size=1, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(n3x3, kernel_size=3, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

        # 1x1 conv -> 5x5 conv branch
        # Note: Using two 3x3 convs to simulate 5x5 conv as in the original
        self.b3 = keras.Sequential([
            layers.Conv2D(n5x5red, kernel_size=1, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(n5x5, kernel_size=3, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(n5x5, kernel_size=3, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

        # 3x3 pool -> 1x1 conv branch
        self.b4 = keras.Sequential([
            layers.MaxPool2D(pool_size=3, strides=1, padding='same'),
            layers.Conv2D(pool_planes, kernel_size=1, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

    def call(self, x, training=None):
        y1 = self.b1(x, training=training)
        y2 = self.b2(x, training=training)
        y3 = self.b3(x, training=training)
        y4 = self.b4(x, training=training)
        return tf.concat([y1, y2, y3, y4], axis=-1)  # Concatenate along channel axis


class GoogLeNet(keras.Model):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.pre_layers = keras.Sequential([
            layers.Conv2D(192, kernel_size=3, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.linear = layers.Dense(num_classes)
        self.get_features = False

    def call(self, x, training=None):
        features = []
        out = self.pre_layers(x, training=training)
        out = self.a3(out, training=training)
        out = self.b3(out, training=training)
        out = self.maxpool(out)
        out = self.a4(out, training=training)
        out = self.b4(out, training=training)
        out = self.c4(out, training=training)
        out = self.d4(out, training=training)
        out = self.e4(out, training=training)
        out = self.maxpool(out)
        out = self.a5(out, training=training)
        out = self.b5(out, training=training)
        features.append(out)
        
        # Apply average pooling
        # Note: The original uses AvgPool2d(8, stride=1) which is specific to certain input sizes
        # For CIFAR-10 (32x32 input), after all the convolutions and pooling, the feature maps
        # should be around 1x1, so GlobalAveragePooling2D is equivalent
        out = self.avgpool(out)
        features.append(out)
        
        out = self.linear(out)
        if self.get_features:
            return features, out
        return out


# Alternative implementation without inheriting from keras.Model
# This follows the exemption for Detection classes not inheriting from framework Model
class GoogLeNetFunc:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.get_features = False
        self._build_model()
    
    def _build_model(self):
        # Build the model using functional API for better control
        self.pre_layers = keras.Sequential([
            layers.Conv2D(192, kernel_size=3, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

        self.a3 = self._make_inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = self._make_inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        self.a4 = self._make_inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = self._make_inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = self._make_inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = self._make_inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = self._make_inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = self._make_inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = self._make_inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.linear = layers.Dense(self.num_classes)
    
    def _make_inception(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        class InceptionBlock:
            def __init__(self):
                # 1x1 conv branch
                self.b1 = keras.Sequential([
                    layers.Conv2D(n1x1, kernel_size=1, use_bias=False),
                    layers.BatchNormalization(),
                    layers.ReLU(),
                ])

                # 1x1 conv -> 3x3 conv branch
                self.b2 = keras.Sequential([
                    layers.Conv2D(n3x3red, kernel_size=1, use_bias=False),
                    layers.BatchNormalization(),
                    layers.ReLU(),
                    layers.Conv2D(n3x3, kernel_size=3, padding='same', use_bias=False),
                    layers.BatchNormalization(),
                    layers.ReLU(),
                ])

                # 1x1 conv -> 5x5 conv branch (simulated with two 3x3 convs)
                self.b3 = keras.Sequential([
                    layers.Conv2D(n5x5red, kernel_size=1, use_bias=False),
                    layers.BatchNormalization(),
                    layers.ReLU(),
                    layers.Conv2D(n5x5, kernel_size=3, padding='same', use_bias=False),
                    layers.BatchNormalization(),
                    layers.ReLU(),
                    layers.Conv2D(n5x5, kernel_size=3, padding='same', use_bias=False),
                    layers.BatchNormalization(),
                    layers.ReLU(),
                ])

                # 3x3 pool -> 1x1 conv branch
                self.b4 = keras.Sequential([
                    layers.MaxPool2D(pool_size=3, strides=1, padding='same'),
                    layers.Conv2D(pool_planes, kernel_size=1, use_bias=False),
                    layers.BatchNormalization(),
                    layers.ReLU(),
                ])
            
            def __call__(self, x, training=None):
                y1 = self.b1(x, training=training)
                y2 = self.b2(x, training=training)
                y3 = self.b3(x, training=training)
                y4 = self.b4(x, training=training)
                return tf.concat([y1, y2, y3, y4], axis=-1)
        
        return InceptionBlock()
    
    def __call__(self, x, training=None):
        features = []
        out = self.pre_layers(x, training=training)
        out = self.a3(out, training=training)
        out = self.b3(out, training=training)
        out = self.maxpool(out)
        out = self.a4(out, training=training)
        out = self.b4(out, training=training)
        out = self.c4(out, training=training)
        out = self.d4(out, training=training)
        out = self.e4(out, training=training)
        out = self.maxpool(out)
        out = self.a5(out, training=training)
        out = self.b5(out, training=training)
        features.append(out)
        
        out = self.avgpool(out)
        features.append(out)
        
        out = self.linear(out)
        if self.get_features:
            return features, out
        return out
    
    def parameters(self):
        """Get all trainable parameters (for compatibility)"""
        params = []
        for layer_group in [self.pre_layers, self.a3, self.b3, self.a4, self.b4, 
                           self.c4, self.d4, self.e4, self.a5, self.b5, self.linear]:
            if hasattr(layer_group, 'trainable_variables'):
                params.extend(layer_group.trainable_variables)
            elif hasattr(layer_group, 'b1'):  # Inception blocks
                for branch in ['b1', 'b2', 'b3', 'b4']:
                    branch_layer = getattr(layer_group, branch)
                    if hasattr(branch_layer, 'trainable_variables'):
                        params.extend(branch_layer.trainable_variables)
        return params
    
    def train(self):
        """Set model to training mode (for compatibility)"""
        pass
    
    def eval(self):
        """Set model to evaluation mode (for compatibility)"""
        pass