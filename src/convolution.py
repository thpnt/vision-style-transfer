import tensorflow as tf

class EncodeConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same', use_relu=True):
        super(EncodeConvBlock, self).__init__()
        self.use_relu = use_relu
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)
        self.batch_norm = InstanceNormalization()
        if use_relu:
            self.relu = tf.keras.layers.ReLU()
        else:
            self.relu = lambda x: x
            
    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)
            
    def build(self, input_shape):
        self.conv.build(input_shape)
        self.batch_norm.build(self.conv.compute_output_shape(input_shape))
        self.built = True
        
    def call(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
    
class DecoderConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=2, padding='same', use_tanh=False, normalize=True):
        super(DecoderConvBlock, self).__init__()
        self.normalize = normalize
        self.use_tanh = use_tanh
        self.transposed_conv = tf.keras.layers.Conv2DTranspose(
            filters, kernel_size, strides=strides, padding=padding
        )
        self.batch_norm = InstanceNormalization() if normalize else None
        self.use_tanh = use_tanh
        self.activation = (lambda x: 0.5 * (tf.keras.activations.tanh(x) + 1)) if use_tanh else tf.keras.layers.ReLU()
        
    def compute_output_shape(self, input_shape):
        return self.transposed_conv.compute_output_shape(input_shape)
            
    def build(self, input_shape):
        self.transposed_conv.build(input_shape)
        if self.batch_norm:
            self.batch_norm.build(self.transposed_conv.compute_output_shape(input_shape))
        self.built = True

    def call(self, inputs):
        x = self.transposed_conv(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x
    
    
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same'):
        super(ResBlock, self).__init__()
        self.conv1 = EncodeConvBlock(filters, kernel_size, strides, padding)
        self.conv2 = EncodeConvBlock(filters, kernel_size, strides, padding, use_relu=False)
    
    def compute_output_shape(self, input_shape):
        return self.conv2.compute_output_shape(self.conv1.compute_output_shape(input_shape))
    
    def build(self, input_shape):
        self.conv1.build(input_shape)
        self.conv2.build(self.conv1.compute_output_shape(input_shape))
        self.built = True
    
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + inputs
    
    
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)  # Compute mean and variance per instance
        return (inputs - mean) / tf.sqrt(variance + self.epsilon)