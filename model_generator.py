import tensorflow as tf


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def gen_deconv(batch_input, out_channels, separable_conv=True):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(
            batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_conv(batch_input, out_channels, separable_conv=True, strides=(2, 2)):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=strides, padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=strides, padding="same", kernel_initializer=initializer)


def batchnorm(inputs, training):
    return tf.layers.batch_normalization(inputs,
                                         axis=3, epsilon=1e-5, momentum=0.1, training=training,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def create_generator_pix2pix(generator_inputs, generator_outputs_channels, ngf, training=True):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, ngf)
        layers.append(output)

    layer_specs = [
        # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        ngf * 2,
        # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 4,
        # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8,
        # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8,
        ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved, training=training)
            layers.append(output)

    layer_specs = [
        # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),
        # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),
        # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.5),
        # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 8, 0.0),
        # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 4, 0.0),
        # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf * 2, 0.0),
        # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        (ngf, 0.0),
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output, training=training)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def conv_bn_relu(x, filters, training):
    conv = gen_conv(x, filters, strides=(1, 1))
    bn = batchnorm(conv, training)
    return tf.nn.relu(bn)


def down_block(input, ngf, training, padding='same', pool_size=2):
    x = tf.layers.max_pooling2d(input, pool_size, pool_size)
    temp = conv_bn_relu(x, ngf, training)
    bn = batchnorm(gen_conv(temp, ngf, strides=(1, 1)), training=training)
    bn += x
    act = tf.nn.relu(bn)
    print(act.shape)
    return bn, act


def up_block(act, bn, training, ngf, use_drop=False):
    bn_shape = tf.shape(bn)
    h, w = bn_shape[1], bn_shape[2]  # bn.get_shape().as_list()[1:3]
    #h *= 2
    #w *= 2
    x = tf.image.resize_images(
        act,
        (h, w),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        align_corners=False
    )
    temp = tf.concat([bn, x], axis=-1)
    temp = conv_bn_relu(temp, ngf, training)
    bn = batchnorm(gen_conv(temp, ngf, strides=(1, 1)), training=training)
    output = tf.nn.relu(bn)
    if use_drop:
        output = tf.nn.dropout(output, keep_prob=0.5)
    print(output.shape)
    return output


def create_generator_deepunet(generator_inputs, generator_outputs_channels, ngf, training=True):
    x = conv_bn_relu(generator_inputs, ngf, training)
    print(generator_inputs.shape)
    net = conv_bn_relu(x, ngf, training)
    bn1 = batchnorm(gen_conv(net, ngf, strides=(1, 1)), training)
    act1 = tf.nn.relu(bn1)
    bn2, act2 = down_block(act1, ngf, training, pool_size=4)
    bn3, act3 = down_block(act2, ngf, training, pool_size=4)
    bn4, act4 = down_block(act3, ngf, training, pool_size=2)
    bn5, act5 = down_block(act4, ngf, training, pool_size=2)
    bn6, act6 = down_block(act5, ngf, training, pool_size=2)

    print('Act6:{}\t, bn5:{}'.format(act6.shape, bn5.shape))
    temp = up_block(act5, bn4, training, ngf, use_drop=True)
    temp = up_block(temp, bn3, training, ngf, use_drop=True)
    temp = up_block(temp, bn2, training, ngf, use_drop=True)

    temp = up_block(temp, bn3, training, ngf)

    temp = up_block(temp, bn2, training, ngf)

    temp = up_block(temp, bn1, training, ngf)
    score1 = tf.tanh(gen_conv(temp, 3, strides=(1, 1)))
    print(score1.shape)
    return score1


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, [None, 512, 512, 3])
    targets = tf.placeholder(tf.float32, [None, 512, 512, 3])
    outputs = create_generator_deepunet(inputs, 3, ngf=16, training=True)
