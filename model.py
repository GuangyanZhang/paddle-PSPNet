import paddle
import paddle.fluid as fluid

def validate_padding(padding, name = ''):
    assert padding in ('SAME', 'VALID')

def conv(x, kernel_size, channel_num, stride_size = 1, padding = 'SAME', name = ''):
    validate_padding(padding)
    if padding == 'SAME':
        padding_size = (kernel_size - 1) / 2
    else:
        padding_size = 0
    return  paddle.fluid.layers.conv2d(x, channel_num, kernel_size, stride = stride_size, padding = padding_size, bias_attr = None, name = name)

def atrous_conv(x, kernel_size, channel_num, dilation, padding = 'SAME', name = ''):
    validate_padding(padding)
    if padding == 'SAME':
        padding_size = ((kernel_size - 1) / 2) + (dilation - 1)
    else:
        padding_size = 0
    return  paddle.fluid.layers.conv2d(x, channel_num, kernel_size, padding = padding_size, bias_attr = None, name = name)

def relu(x, name = ''):
    return paddle.fluid.layers.relu(x)

def max_pool(x, kernel_size, stride_size, padding = 'SAME', name = ''):
    validate_padding(padding)
    if padding == 'SAME':
        padding_size = (kernel_size - 1) / 2
    else:
        padding_size = 0
    return paddle.fluid.layers.pool2d(x, pool_size = kernel_size, pool_stride = stride_size, pool_padding = padding_size, name = name)
    
def avg_pool(x, kernel_size, stride_size, padding = 'VALID', name = ''):
    validate_padding(padding)
    if padding == 'SAME':
        padding_size = (kernel_size - 1) / 2
    else:
        padding_size = 0
    return paddle.fluid.layers.pool2d(x, pool_size = kernel_size, pool_stride = stride_size, pool_padding = padding_size, name = name)

def concat(inputs, axis = -1, name = ''):
    return paddle.fluid.layers.concat(inputs, axis = axis, name = name)

def add(x, y, with_relu = True, name = ''):
    if with_relu:
        return paddle.fluid.layers.elementwise_add(x, y, act = 'relu', name = name)
    else:
        return paddle.fluid.layers.elementwise_add(x, y, name = name)

def batch_normalization(x, relu = False, name = ''):
    if relu:
        return paddle.fluid.layers.batch_norm(x, act = 'relu', name = name)
    else:
        return paddle.fluid.layers.batch_norm(x, name = name)

def get_shape(x):
    return paddle.fluid.layers.shape(x)[3:4]

def resize_bilinear(x, shape, name = ''):
    return paddle.fluid.layers.resize_bilinear(x, shape, name = name)