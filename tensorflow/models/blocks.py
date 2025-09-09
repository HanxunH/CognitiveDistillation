import tensorflow as tf
from tensorflow.keras import layers


class Conv2dBlock(tf.keras.Model):
    def __init__(self, in_c, out_c, ker_size=(3, 3), stride=1, padding=1, batch_norm=True, relu=True):
        super(Conv2dBlock, self).__init__()
        
        # TensorFlow uses 'valid' or 'same' padding, need to handle explicit padding
        # For explicit padding values, we'll use a ZeroPadding2D layer
        self.explicit_padding = None
        if isinstance(padding, int) and padding > 0:
            self.explicit_padding = layers.ZeroPadding2D(padding=padding)
            padding_mode = 'valid'
        else:
            padding_mode = 'same' if padding else 'valid'
        
        self.conv2d = layers.Conv2D(
            filters=out_c,
            kernel_size=ker_size,
            strides=stride,
            padding=padding_mode,
            use_bias=True
        )
        
        self.batch_norm = None
        if batch_norm:
            # TensorFlow BatchNorm momentum is opposite of PyTorch
            # PyTorch momentum=0.05 means 5% current, 95% running
            # TensorFlow momentum=0.95 means 95% running, 5% current
            self.batch_norm = layers.BatchNormalization(
                epsilon=1e-5,
                momentum=0.95,  # 1 - 0.05 from PyTorch
                center=True,
                scale=True
            )
        
        self.relu = None
        if relu:
            # TensorFlow ReLU doesn't have inplace parameter (exemption #6)
            self.relu = layers.ReLU()
    
    def call(self, x, training=None):
        if self.explicit_padding is not None:
            x = self.explicit_padding(x)
        
        x = self.conv2d(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x, training=training)
        
        if self.relu is not None:
            x = self.relu(x)
        
        return x


class DownSampleBlock(tf.keras.Model):
    def __init__(self, ker_size=(2, 2), stride=2, dilation=(1, 1), ceil_mode=False, p=0.0):
        super(DownSampleBlock, self).__init__()
        
        # MaxPool2D in TensorFlow doesn't support dilation parameter (exemption #11)
        # If dilation is not default (1,1), we need to handle it differently
        if dilation != (1, 1) and dilation != 1:
            raise NotImplementedError(
                f"TensorFlow MaxPool2D doesn't support dilation={dilation}. "
                "Only dilation=(1,1) is supported."
            )
        
        # TensorFlow uses 'valid' or 'same' padding
        # ceil_mode in PyTorch affects how output size is calculated with padding
        # In TensorFlow, we can approximate this with padding='same' when ceil_mode=True
        padding = 'same' if ceil_mode else 'valid'
        
        self.maxpooling = layers.MaxPool2D(
            pool_size=ker_size,
            strides=stride,
            padding=padding
        )
        
        self.dropout = None
        if p:
            self.dropout = layers.Dropout(rate=p)
    
    def call(self, x, training=None):
        x = self.maxpooling(x)
        
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        
        return x


class UpSampleBlock(tf.keras.Model):
    def __init__(self, scale_factor=(2, 2), mode="bilinear", p=0.0):
        super(UpSampleBlock, self).__init__()
        
        # Map PyTorch interpolation modes to TensorFlow
        if mode == "bilinear":
            interpolation = "bilinear"
        elif mode == "nearest":
            interpolation = "nearest"
        else:
            interpolation = mode
        
        self.upsample = layers.UpSampling2D(
            size=scale_factor,
            interpolation=interpolation
        )
        
        self.dropout = None
        if p:
            self.dropout = layers.Dropout(rate=p)
    
    def call(self, x, training=None):
        x = self.upsample(x)
        
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        
        return x