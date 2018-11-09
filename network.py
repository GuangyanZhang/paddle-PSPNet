from model import *


def block_conv1(layer):
    layer = conv(layer, 3, 64, 2, padding='SAME', name='conv1_0_3x3_s2')
    layer = batch_normalization(layer, relu=False, name='conv1_0_3x3_s2_bn')
    layer = relu(layer, name='conv1_0_3x3_s2_bn_relu')
    layer = conv(layer, 3, 64, 1, padding='SAME', name='conv1_1_3x3')
    layer = batch_normalization(layer, relu=True, name='conv1_1_3x3_bn')
    layer = conv(layer, 3, 128, 1, padding='SAME', name='conv1_2_3x3')
    layer = batch_normalization(layer, relu=True, name='conv1_2_3x3_bn')
    layer = max_pool(layer, 3, 2, padding='SAME', name='pool1_3x3_s2')
    return layer


def conv1(layer):
    layer = block_conv1(layer)
    return layer


def block_conv2(layer, idx):
    name = 'conv2_' + str(idx)
    layer = conv(layer, 1, 64, 1, name=name + '_1x1_reduce')
    layer = batch_normalization(layer, relu=True, name=name + '_1x1_reduce_bn')
    layer = conv(layer, 3, 64, 1, name=name + '_3x3')
    layer = batch_normalization(layer, relu=True, name=name + '_3x3_bn')
    layer = conv(layer, 1, 256, 1, name=name + '_1x1_increase')
    layer = batch_normalization(layer, relu=False, name=name + '_1x1_increase_bn')
    return layer


def conv2(x, num):
    layer = x
    layer = conv(layer, 1, 256, 1, name='conv2_1_1x1_proj')
    layer = batch_normalization(layer, relu=False, name='conv2_1_1x1_proj_bn')
    last_layer = layer
    layer = x

    for i in range(num):
        layer = block_conv2(layer, i)
        layer = add(last_layer, layer, name='conv2_'+str(i))
        last_layer = layer

    return layer


def block_conv3(layer, idx):
    name = 'conv3_' + str(idx)
    layer = conv(layer ,1, 128, 2, name=name + '_1x1_reduce')
    layer = batch_normalization(layer, relu=True, name=name + '_1x1_reduce_bn')
    layer = conv(layer, 3, 128, 1, name=name + '_3x3')
    layer = batch_normalization(layer, relu=True, name=name + '_3x3_bn')
    layer = conv(layer, 1, 512, 1, name=name + '_1x1_increase')
    layer = batch_normalization(layer, relu=False, name=name + '_1x1_increase_bn')
    return layer


def conv3(x, num):
    layer = x
    layer = conv(layer, 1, 512, 2, name='conv3_1_1x1_proj')
    layer = batch_normalization(layer, relu=False, name='conv3_1_1x1_proj_bn')
    last_layer = layer
    layer = x

    for i in range(num):
        layer = block_conv3(layer, i)
        layer = add(last_layer, layer, name='conv3_'+str(i))
        last_layer = layer
    
    return layer


def block_conv4(layer, idx):
    name = 'conv4_' + str(idx)
    layer = conv(layer, 1, 256, 1, name=name + '_1x1_reduce')
    layer = batch_normalization(layer, relu=True, name=name + '_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name=name + '_3x3')
    layer = batch_normalization(layer, relu=True, name=name + '_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name=name + '_1x1_increase')
    layer = batch_normalization(layer, relu=False, name=name + '_1x1_increase_bn')
    return layer

def conv4(x, num):
    layer = x
    layer = conv(layer, 1, 1024, 1, name='conv4_1_1x1_proj')
    layer = batch_normalization(layer, relu=False, name='conv4_1_1x1_proj_bn')
    last_layer = layer
    layer = x

    for i in range(num):
        layer = block_conv4(layer, i)
        layer = add(last_layer, layer, name='conv4_'+str(i))
        last_layer = layer
    
    return layer


def block_conv5(layer, idx):
    name = 'conv5_' + str(idx)
    layer = conv(layer, 1, 512, 1, name=name + '_1x1_reduce')
    layer = batch_normalization(layer, relu=True, name=name + '_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 512, 4, name=name + '_3x3')
    layer = batch_normalization(layer, relu=True, name=name + '_3x3_bn')
    layer = conv(layer, 1, 2048, 1, name=name + '_1x1_increase')
    layer = batch_normalization(layer, relu=False, name=name + '_1x1_increase_bn')
    return layer


def conv5(x, num):
    layer = x
    layer = conv(layer, 1, 2048, 1, name='conv5_1_1x1_proj')
    layer = batch_normalization(layer, relu=False, name='conv5_1_1x1_proj_bn')
    last_layer = layer
    layer = x

    for i in range(num):
        layer = block_conv5(layer, i)
        layer = add(last_layer, layer, name='conv5_'+str(i))
        last_layer = layer
    
    return layer


def block_conv5_3_pool(layer, shape, pool_size, idx):
    name = 'conv5_3_pool' + str(idx)
    layer = avg_pool(layer, pool_size, pool_size, name=name + '')
    layer = conv(layer, 1, 512, 1, name=name + '_conv')
    layer = batch_normalization(layer, relu=True, name=name + '_conv_bn')
    layer = resize_bilinear(layer, shape, name=name + '_interp')
    return layer


def conv5_3_pool(x):
    shape = get_shape(x)

    conv5_3_pool1_interp = block_conv5_3_pool(x, shape, 90, 1)
    conv5_3_pool2_interp = block_conv5_3_pool(x, shape, 45, 2)
    conv5_3_pool3_interp = block_conv5_3_pool(x, shape, 30, 3)
    conv5_3_pool6_interp = block_conv5_3_pool(x, shape, 15, 6)

    group = [x, conv5_3_pool6_interp, conv5_3_pool3_interp, conv5_3_pool2_interp, conv5_3_pool1_interp]
    layer = concat(group, axis=1, name='conv5_3_concat')
    return layer


def PSPNet101(x, num_classes):
    layer = x
    layer = conv1(layer)
    layer = conv2(layer, 3)
    layer = conv3(layer, 4)
    layer = conv4(layer, 23)
    layer = conv5(layer, 3)

    layer = conv5_3_pool(layer)
    layer = conv(layer, 3, 512, 1, padding='SAME', name='conv5_4')
    layer = batch_normalization(layer, relu=True, name='conv5_4_bn')
    layer = conv(layer, 1, num_classes, 1, name='conv6')
    return layer


def PSPNet50(x, num_classes):
    layer = x
    layer = conv1(layer)
    layer = conv2(layer, 3)
    layer = conv3(layer, 4)
    layer = conv4(layer, 6)
    layer = conv5(layer, 3)

    layer = conv5_3_pool(layer)
    layer = conv(layer, 3, 512, 1, padding='SAME', name='conv5_4')
    layer = batch_normalization(layer, relu=True, name='conv5_4_bn')
    layer = conv(layer, 1, num_classes, 1, name='conv6')
    return layer