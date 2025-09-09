import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class BasicBlock(Model):
    """Basic residual block for ResNet18/34."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(planes, kernel_size=3,
                                   strides=stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(planes, kernel_size=3, strides=1,
                                   padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential([
                layers.Conv2D(self.expansion * planes, kernel_size=1,
                              strides=stride, use_bias=False),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x  # Identity function

    def call(self, x, training=None):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        out += self.shortcut(x)
        out = tf.nn.relu(out)
        return out


class ResNet18_200(Model):
    """ResNet18 for ISSBA with 200 classes (ImageNet-style architecture).
    
    This model follows the standard ImageNet ResNet18 architecture with:
    - Initial 7x7 conv with stride 2
    - BatchNorm and ReLU
    - 3x3 MaxPool with stride 2
    - 4 residual stages
    - Global average pooling
    - FC layer for 200 classes
    """
    
    def __init__(self, num_classes=200):
        super(ResNet18_200, self).__init__()
        self.get_features = False
        self.in_planes = 64
        
        # Initial layers (ImageNet-style)
        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, 
                                   padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        
        # Residual layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # Final layers
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        """Create a residual layer with multiple blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers_list = []
        for stride in strides:
            layers_list.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential(layers_list)
    
    def call(self, x, training=None):
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            training: Boolean or None. If True, network runs in training mode.
        
        Returns:
            If self.get_features is True: (features, logits)
            Otherwise: logits
            
            Where features is a list containing:
            - Feature map after layer4
            - Flattened features after global pooling
        """
        features = []
        
        # Initial convolution layers
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        
        # Store feature map after layer4 (before pooling)
        features.append(x)
        
        # Global average pooling
        x = self.avgpool(x)
        
        # Store flattened features
        features.append(x)
        
        # Final classification layer
        x = self.fc(x)
        
        if self.get_features:
            return features, x
        return x