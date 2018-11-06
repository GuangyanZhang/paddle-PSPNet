from m import *

def PSPNet101(x, num_classes):
    layer = x
    layer = conv(layer, 3, 64, 2, padding='SAME', name='conv1_1_3x3_s2')
    layer = batch_normalization(relu=False, name='conv1_1_3x3_s2_bn')
    layer = relu(name='conv1_1_3x3_s2_bn_relu')
    layer = conv(layer, 3, 64, 1, padding='SAME', name='conv1_2_3x3')
    layer = batch_normalization(relu=True, name='conv1_2_3x3_bn')
    layer = conv(layer, 3, 128, 1, padding='SAME', name='conv1_3_3x3')
    layer = batch_normalization(relu=True, name='conv1_3_3x3_bn')
    layer = max_pool(3, 2, padding='SAME', name='pool1_3x3_s2')
    pool1_3x3_s2 = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv2_1_1x1_proj')
    layer = batch_normalization(relu=False, name='conv2_1_1x1_proj_bn')
    conv2_1_1x1_proj_bn = layer #####################

    layer = conv(pool1_3x3_s2, 1, 64, 1, name='conv2_1_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv2_1_1x1_reduce_bn')
    layer = conv(layer, 3, 64, 1, name='conv2_1_3x3')
    layer = batch_normalization(relu=True, name='conv2_1_3x3_bn')
    layer = conv(layer, 1, 256, 1, name='conv2_1_1x1_increase')
    layer = batch_normalization(relu=False, name='conv2_1_1x1_increase_bn')
    conv2_1_1x1_increase_bn = layer #####################

    layer = add(conv2_1_1x1_proj_bn, conv2_1_1x1_increase_bn, name='conv2_1')
    layer = relu(name='conv2_1_relu')
    conv2_1_relu = layer #####################
    layer = conv(layer, 1, 64, 1, name='conv2_2_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv2_2_1x1_reduce_bn')
    layer = conv(layer, 3, 64, 1, name='conv2_2_3x3')
    layer = batch_normalization(relu=True, name='conv2_2_3x3_bn')
    layer = conv(layer, 1, 256, 1, name='conv2_2_1x1_increase')
    layer = batch_normalization(relu=False, name='conv2_2_1x1_increase_bn')
    conv2_2_1x1_increase_bn = layer #####################

    layer = add(conv2_1_relu, conv2_2_1x1_increase_bn, name='conv2_2')
    layer = relu(name='conv2_2_relu')
    conv2_2_relu = layer #####################
    layer = conv(layer, 1, 64, 1, name='conv2_3_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv2_3_1x1_reduce_bn')
    layer = conv(layer, 3, 64, 1, name='conv2_3_3x3')
    layer = batch_normalization(relu=True, name='conv2_3_3x3_bn')
    layer = conv(layer, 1, 256, 1, name='conv2_3_1x1_increase')
    layer = batch_normalization(relu=False, name='conv2_3_1x1_increase_bn')
    conv2_3_1x1_increase_bn = layer #####################

    layer = add(conv2_2_relu, conv2_3_1x1_increase_bn, name='conv2_3')
    layer = relu(name='conv2_3_relu')
    conv2_3_relu = layer #####################
    layer = conv(layer, 1, 512, 2, name='conv3_1_1x1_proj')
    layer = batch_normalization(relu=False, name='conv3_1_1x1_proj_bn')
    conv3_1_1x1_proj_bn = layer #####################

    layer = conv(conv2_3_relu ,1, 128, 2, name='conv3_1_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv3_1_1x1_reduce_bn')
    layer = conv(layer, 3, 128, 1, name='conv3_1_3x3')
    layer = batch_normalization(relu=True, name='conv3_1_3x3_bn')
    layer = conv(layer, 1, 512, 1, name='conv3_1_1x1_increase')
    layer = batch_normalization(relu=False, name='conv3_1_1x1_increase_bn')
    conv3_1_1x1_increase_bn = layer #####################

    layer = add(conv3_1_1x1_proj_bn, conv3_1_1x1_increase_bn, name='conv3_1')
    layer = relu(name='conv3_1_relu')
    conv3_1_relu = layer #####################
    layer = conv(layer, 1, 128, 1, name='conv3_2_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv3_2_1x1_reduce_bn')
    layer = conv(layer, 3, 128, 1, name='conv3_2_3x3')
    layer = batch_normalization(relu=True, name='conv3_2_3x3_bn')
    layer = conv(layer, 1, 512, 1, name='conv3_2_1x1_increase')
    layer = batch_normalization(relu=False, name='conv3_2_1x1_increase_bn')
    conv3_2_1x1_increase_bn = layer #####################

    layer = add(conv3_1_relu, conv3_2_1x1_increase_bn, name='conv3_2')
    layer = relu(name='conv3_2_relu')
    conv3_2_relu = layer #####################
    layer = conv(layer, 1, 128, 1, name='conv3_3_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv3_3_1x1_reduce_bn')
    layer = conv(layer, 3, 128, 1, name='conv3_3_3x3')
    layer = batch_normalization(relu=True, name='conv3_3_3x3_bn')
    layer = conv(layer, 1, 512, 1, name='conv3_3_1x1_increase')
    layer = batch_normalization(relu=False, name='conv3_3_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################




#####################################################################################
    layer = add(conv3_2_relu, conv3_3_1x1_increase_bn, name='conv3_3')
    layer = relu(name='conv3_3_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 128, 1, name='conv3_4_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv3_4_1x1_reduce_bn')
    layer = conv(layer, 3, 128, 1, name='conv3_4_3x3')
    layer = batch_normalization(relu=True, name='conv3_4_3x3_bn')
    layer = conv(layer, 1, 512, 1, name='conv3_4_1x1_increase')
    layer = batch_normalization(relu=False, name='conv3_4_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv3_3_relu
    conv3_4_1x1_increase_bn
    layer = add(name='conv3_4')
    layer = relu(name='conv3_4_relu')
    layer = conv(layer, 1, 1024, 1, name='conv4_1_1x1_proj')
    layer = batch_normalization(relu=False, name='conv4_1_1x1_proj_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv3_4_relu
    layer = conv(layer, 1, 256, 1, name='conv4_1_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_1_3x3')
    layer = batch_normalization(relu=True, name='conv4_1_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_1_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_1_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_1_1x1_proj_bn',
    conv4_1_1x1_increase_bn')
    layer = add(name='conv4_1')
    layer = relu(name='conv4_1_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_2_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn')\
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_2_3x3')
    layer = batch_normalization(relu=True, name='conv4_2_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_2_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_2_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_1_relu',
    conv4_2_1x1_increase_bn')
    layer = add(name='conv4_2')
    layer = relu(name='conv4_2_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_3_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_3_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_3_3x3')
    layer = batch_normalization(relu=True, name='conv4_3_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_3_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_3_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_2_relu',
    conv4_3_1x1_increase_bn')
    layer = add(name='conv4_3')
    layer = relu(name='conv4_3_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_4_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_4_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_4_3x3')
    layer = batch_normalization(relu=True, name='conv4_4_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_4_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_4_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_3_relu',
    conv4_4_1x1_increase_bn')
    layer = add(name='conv4_4')
    layer = relu(name='conv4_4_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_5_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_5_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_5_3x3')
    layer = batch_normalization(relu=True, name='conv4_5_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_5_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_5_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_4_relu',
    conv4_5_1x1_increase_bn')
    layer = add(name='conv4_5')
    layer = relu(name='conv4_5_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_6_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_6_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_6_3x3')
    layer = batch_normalization(relu=True, name='conv4_6_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_6_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_6_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_5_relu',
    conv4_6_1x1_increase_bn')
    layer = add(name='conv4_6')
    layer = relu(name='conv4_6_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_7_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_7_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_7_3x3')
    layer = batch_normalization(relu=True, name='conv4_7_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_7_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_7_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_6_relu',
    conv4_7_1x1_increase_bn')
    layer = add(name='conv4_7')
    layer = relu(name='conv4_7_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_8_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_8_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_8_3x3')
    layer = batch_normalization(relu=True, name='conv4_8_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_8_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_8_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_7_relu',
    conv4_8_1x1_increase_bn')
    layer = add(name='conv4_8')
    layer = relu(name='conv4_8_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_9_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_9_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_9_3x3')
    layer = batch_normalization(relu=True, name='conv4_9_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_9_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_9_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_8_relu',
    conv4_9_1x1_increase_bn')
    layer = add(name='conv4_9')
    layer = relu(name='conv4_9_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_10_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_10_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_10_3x3')
    layer = batch_normalization(relu=True, name='conv4_10_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_10_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_10_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_9_relu',
    conv4_10_1x1_increase_bn')
    layer = add(name='conv4_10')
    layer = relu(name='conv4_10_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_11_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_11_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_11_3x3')
    layer = batch_normalization(relu=True, name='conv4_11_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_11_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_11_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_10_relu',
    conv4_11_1x1_increase_bn')
    layer = add(name='conv4_11')
    layer = relu(name='conv4_11_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_12_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_12_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_12_3x3')
    layer = batch_normalization(relu=True, name='conv4_12_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_12_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_12_1x1_increase_bn')

    conv4_11_relu',
    conv4_12_1x1_increase_bn')
    layer = add(name='conv4_12')
    layer = relu(name='conv4_12_relu')
    layer = conv(layer, 1, 256, 1, name='conv4_13_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_13_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_13_3x3')
    layer = batch_normalization(relu=True, name='conv4_13_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_13_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_13_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_12_relu',
    conv4_13_1x1_increase_bn')
    layer = add(name='conv4_13')
    layer = relu(name='conv4_13_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_14_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_14_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_14_3x3')
    layer = batch_normalization(relu=True, name='conv4_14_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_14_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_14_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_13_relu',
    conv4_14_1x1_increase_bn')
    layer = add(name='conv4_14')
    layer = relu(name='conv4_14_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_15_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_15_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_15_3x3')
    layer = batch_normalization(relu=True, name='conv4_15_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_15_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_15_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_14_relu',
    conv4_15_1x1_increase_bn')
    layer = add(name='conv4_15')
    layer = relu(name='conv4_15_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_16_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_16_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_16_3x3')
    layer = batch_normalization(relu=True, name='conv4_16_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_16_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_16_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_15_relu',
    conv4_16_1x1_increase_bn')
    layer = add(name='conv4_16')
    layer = relu(name='conv4_16_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_17_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_17_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_17_3x3')
    layer = batch_normalization(relu=True, name='conv4_17_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_17_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_17_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_16_relu',
    conv4_17_1x1_increase_bn')
    layer = add(name='conv4_17')
    layer = relu(name='conv4_17_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_18_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_18_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_18_3x3')
    layer = batch_normalization(relu=True, name='conv4_18_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_18_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_18_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_17_relu',
    conv4_18_1x1_increase_bn')
    layer = add(name='conv4_18')
    layer = relu(name='conv4_18_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_19_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_19_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_19_3x3')
    layer = batch_normalization(relu=True, name='conv4_19_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_19_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_19_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_18_relu',
    conv4_19_1x1_increase_bn')
    layer = add(name='conv4_19')
    layer = relu(name='conv4_19_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_20_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_20_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_20_3x3')
    layer = batch_normalization(relu=True, name='conv4_20_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_20_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_20_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_19_relu',
    conv4_20_1x1_increase_bn')
    layer = add(name='conv4_20')
    layer = relu(name='conv4_20_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_21_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_21_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_21_3x3')
    layer = batch_normalization(relu=True, name='conv4_21_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_21_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_21_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_20_relu',
    conv4_21_1x1_increase_bn')
    layer = add(name='conv4_21')
    layer = relu(name='conv4_21_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_22_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_22_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_22_3x3')
    layer = batch_normalization(relu=True, name='conv4_22_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_22_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_22_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_21_relu',
    conv4_22_1x1_increase_bn')
    layer = add(name='conv4_22')
    layer = relu(name='conv4_22_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 256, 1, name='conv4_23_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv4_23_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 256, 2, name='conv4_23_3x3')
    layer = batch_normalization(relu=True, name='conv4_23_3x3_bn')
    layer = conv(layer, 1, 1024, 1, name='conv4_23_1x1_increase')
    layer = batch_normalization(relu=False, name='conv4_23_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_22_relu',
    conv4_23_1x1_increase_bn')
    layer = add(name='conv4_23')
    layer = relu(name='conv4_23_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 2048, 1, name='conv5_1_1x1_proj')
    layer = batch_normalization(relu=False, name='conv5_1_1x1_proj_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv4_23_relu')
    layer = conv(layer, 1, 512, 1, name='conv5_1_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv5_1_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 512, 4, name='conv5_1_3x3')
    layer = batch_normalization(relu=True, name='conv5_1_3x3_bn')
    layer = conv(layer, 1, 2048, 1, name='conv5_1_1x1_increase')
    layer = batch_normalization(relu=False, name='conv5_1_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv5_1_1x1_proj_bn',
    conv5_1_1x1_increase_bn')
    layer = add(name='conv5_1')
    layer = relu(name='conv5_1_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 512, 1, name='conv5_2_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv5_2_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 512, 4, name='conv5_2_3x3')
    layer = batch_normalization(relu=True, name='conv5_2_3x3_bn')
    layer = conv(layer, 1, 2048, 1, name='conv5_2_1x1_increase')
    layer = batch_normalization(relu=False, name='conv5_2_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv5_1_relu',
    conv5_2_1x1_increase_bn')
    layer = add(name='conv5_2')
    layer = relu(name='conv5_2_relu')
    conv3_3_1x1_increase_bn = layer #####################
    layer = conv(layer, 1, 512, 1, name='conv5_3_1x1_reduce')
    layer = batch_normalization(relu=True, name='conv5_3_1x1_reduce_bn')
    layer = atrous_conv(layer, 3, 512, 4, name='conv5_3_3x3')
    layer = batch_normalization(relu=True, name='conv5_3_3x3_bn')
    layer = conv(layer, 1, 2048, 1, name='conv5_3_1x1_increase')
    layer = batch_normalization(relu=False, name='conv5_3_1x1_increase_bn')
    conv3_3_1x1_increase_bn = layer #####################

    conv5_2_relu',
    conv5_3_1x1_increase_bn')
    layer = add(name='conv5_3')
    layer = relu(name='conv5_3_relu')
    conv3_3_1x1_increase_bn = layer #####################

        conv5_3 = self.layers['conv5_3_relu']
        shape = tf.shape(conv5_3)[1:3]

    conv5_3_relu')
    layer = avg_pool(90, 90, 90, 90, name='conv5_3_pool1')
    layer = conv(layer, 1, 512, 1, name='conv5_3_pool1_conv')
    layer = batch_normalization(relu=True, name='conv5_3_pool1_conv_bn')
    layer = resize_bilinear(shape, name='conv5_3_pool1_interp')
    conv3_3_1x1_increase_bn = layer #####################

    conv5_3_relu')
    layer = avg_pool(45, 45, 45, 45, name='conv5_3_pool2')
    layer = conv(layer, 1, 512, 1, name='conv5_3_pool2_conv')
    layer = batch_normalization(relu=True, name='conv5_3_pool2_conv_bn')
    layer = resize_bilinear(shape, name='conv5_3_pool2_interp')
    conv3_3_1x1_increase_bn = layer #####################

    conv5_3_relu')
    layer = avg_pool(30, 30, 30, 30, name='conv5_3_pool3')
    layer = conv(layer, 1, 512, 1, name='conv5_3_pool3_conv')
    layer = batch_normalization(relu=True, name='conv5_3_pool3_conv_bn')
    layer = resize_bilinear(shape, name='conv5_3_pool3_interp')
    conv3_3_1x1_increase_bn = layer #####################

    conv5_3_relu')
    layer = avg_pool(15, 15, 15, 15, name='conv5_3_pool6')
    layer = conv(layer, 1, 512, 1, name='conv5_3_pool6_conv')
    layer = batch_normalization(relu=True, name='conv5_3_pool6_conv_bn')
    layer = resize_bilinear(shape, name='conv5_3_pool6_interp')
    conv3_3_1x1_increase_bn = layer #####################

    conv5_3_relu',
    conv5_3_pool6_interp',
    conv5_3_pool3_interp',
    conv5_3_pool2_interp',
    conv5_3_pool1_interp')
    layer = concat(axis=-1, name='conv5_3_concat')
    layer = conv(layer, 3, 512, 1, padding='SAME', name='conv5_4')
    layer = batch_normalization(relu=True, name='conv5_4_bn')
    layer = conv(layer, 1, num_classes, 1, biased=True, relu=False, name='conv6')
    return layer