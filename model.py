import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np


def CBAM(X, reduction=16):
    """
    @Convolutional Block Attention Module
    """

    imput_shape = X.shape  # (B, W, H, C)
    batch, height, width, channel = imput_shape[0],imput_shape[1],imput_shape[2],imput_shape[3]
    #shared MLP
    shared_layer_one = layers.Dense(channel//reduction,
                                    activation=tf.nn.relu,kernel_initializer='he_normal',
                                    use_bias=True,bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel,kernel_initializer='he_normal',
                                    use_bias=True,bias_initializer='zeros')
    # channel attention
    avg_pool = layers.GlobalAveragePooling2D()(X)
    avg_pool = layers.Reshape([1,1,channel])(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(X)
    max_pool = layers.Reshape([1,1,channel])(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    x = tf.add(avg_pool, max_pool)   # (B, 1, 1, C)
    x = tf.nn.sigmoid(x)        # (B, 1, 1, C)
    x = tf.multiply(X, x)   # (B, W, H, C)

    # spatial attention
    y_mean = tf.reduce_mean(x, axis=3, keepdims=True)  # (B, W, H, 1)
    y_max = tf.reduce_max(x, axis=3, keepdims=True)  # (B, W, H, 1)
    y = tf.concat([y_mean, y_max], axis=3) # (B, W, H, 2)
    y = layers.Conv2D(filters=1, kernel_size=7, padding='same', activation=tf.nn.sigmoid)(y)    # (B, W, H, 1)
    y = tf.multiply(x, y)  # (B, W, H, C)

    return y


def SKConv(M=2, r=16, L=32, G=32,name='skconv'):
    # """
    # L: the minimum dim of the vector z in paper, default 32.
    # r: the radio for compute d, the length of z.
    # M: the number of branchs.
    # G: num of convolution groups.
    # """
    def wrapper(inputs):
        inputs_shape = inputs.shape
        b, h, w = inputs_shape[0], inputs_shape[1], inputs_shape[2]
        filters = inputs.get_shape().as_list()[-1]  
        d = max(int(filters//r), L)  #reduced feature dimension

        x = inputs

        xs = []
        #split
       
        for m in range(1, M+1):
            if G == 1:
                _x = layers.Conv2D(filters, 3, dilation_rate=m, padding='same',
                          use_bias=False, name=name+'_conv%d'%m)(x)
            else:
                c = filters // G
                _x = layers.DepthwiseConv2D(3, dilation_rate=m, depth_multiplier=c, padding='same',
                                   use_bias=False, name=name+'_conv%d'%m)(x)
                
                _x = layers.Reshape([h, w, G, c, c], name=name+'_conv%d_reshape1'%m)(_x)

                _x = tf.reduce_sum(_x, axis=-1)

                _x = layers.Reshape([h, w, filters],
                           name=name+'_conv%d_reshape2'%m)(_x)

            _x = layers.BatchNormalization(name=name+'_conv%d_bn'%m)(_x)
            _x = layers.Activation('relu', name=name+'_conv%d_relu'%m)(_x)

            xs.append(_x)

        U = layers.Add(name=name+'_add')(xs)
        s = tf.reduce_mean(U, axis=[1,2], keepdims=True)

        z = layers.Conv2D(d, 1, name=name+'_fc_z')(s)
        z = layers.BatchNormalization(name=name+'_fc_z_bn')(z)
        z = layers.Activation('relu', name=name+'_fc_z_relu')(z)

        x = layers.Conv2D(filters*M, 1, name=name+'_fc_x')(z)
        x = layers.Reshape([1, 1, filters, M],name=name+'_reshape')(x)
        scale = layers.Softmax(name=name+'_softmax')(x)

        x = tf.stack(xs, axis=-1)

        x = Axpby(name=name+'_axpby')([scale, x])
        return x
    return wrapper


class Axpby(Layer):
    """
    Do this:
        F = a * X + b * Y + ...
        Shape info:
        a:  B x 1 x 1 x C
        X:  B x H x W x C
        b:  B x 1 x 1 x C
        Y:  B x H x W x C
        ...
        F:  B x H x W x C

        weight和矩陣相乘
    """
    def __init__(self, **kwargs):
        super(Axpby, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Axpby, self).build(input_shape)  

    def call(self, inputs):
        """ scale: [B, 1, 1, C, M]
            x: [B, H, W, C, M]
        """
        scale, x = inputs
        f = tf.multiply(scale, x, name='product')
        f = tf.reduce_sum(f, axis=-1, name='sum')
        return f

    def compute_output_shape(self, input_shape):
        return input_shape[0:4]

def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def build_model(batch):

    num_filters = [64, 128 ,256,512]
    inputs = Input((176, 176, 1),batch_size=batch, dtype='float', name='data')

    skip_x = []
    x = inputs

    ## Encoder
    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)

    ## Bridge
    x = SKConv(M=2, r=16, L=32, G=32, name='skconv'+'mid')(x)

    #sknet (copy and crop)
    for i in range(3):
       skip_x[i]=SKConv(M=2, r=16, L=32, G=32, name='skconv'+str(i))(skip_x[i])

    skip_x.reverse()
    num_filters.reverse()

    ## Decoder
    for i, f in enumerate(num_filters): 
        x = Conv2D(f, 1, padding='same', activation='relu')(UpSampling2D(size=[2, 2])(x))
        xs = skip_x[i]
        x = Concatenate()([xs, x])
        x = CBAM(x, reduction=16)
        x = conv_block(x, f)

    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)

if __name__ == "__main__":
    model = build_model(batch=16)
    model.summary()
